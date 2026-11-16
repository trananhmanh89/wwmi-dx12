// matcap_dxil.inc -- OURS. Included by rabbitfx\rfx_dxil.cpp, inside its anonymous namespace, after Cut(): it uses
// that file's Module text helpers. The DXIL form of the KatFXMatcap mod's [ShaderRegex...Matcap] patch.
//
// On DX11 the regex finds the world normal register the character pixel shader already computed, redirects the
// writes of o0 (scene colour) and o3 (base colour) into temporaries, and at the end blends up to two matcap
// textures (t125, t124, optional masks t126, t127) into them. The DX12 game shader has the same eight outputs:
//   SV_Target0.rgb  scene colour (the "emissive" the matcap adds to)
//   SV_Target1.xy   the world normal, octahedral: 0.5 * oct + 0.5 (UE4 UnitVectorToOctahedron)
//   SV_Target3.rgb  base colour
// and cb0 is the engine's View buffer (type %hostlayout.View, 6016 bytes) whose rows 12..14 are the
// translated-world-to-view rotation, the cb0[12..14] the ini reads.
//
// Instead of finding the shader's normal register, the patch decodes SV_Target1.xy again from the very values the
// shader stores there: the stored encoding is the one thing every body shader has in the same form, and it is the
// normal the game lights with. The blend then runs straight-line (selects, no branches) right before the stores.
// The ini's parameters come from a Buffer<float4> of two texels, [0] = x125..w125, [1] = x126..w126:
//   x125 / z125 blend mode of layer 1 / 2       y125 / w125 opacity
//   x126 / y126 strength                        z126 / w126 mask opacity
// Blend modes: 0 lerp towards the matcap in the scene colour, 1 add to it, 2 screen, 4 lerp the base colour,
// anything else multiplies the base colour.

// One fast-math instruction per line, each result named %mc_N.
struct McIr {
    std::string text;
    int next = 0;

    std::string Def(const std::string& rhs)
    {
        const std::string name = "%mc_" + std::to_string(next++);
        text += "  " + name + " = " + rhs + "\n";
        return name;
    }
    std::string Op(const char* op, const std::string& a, const std::string& b)
    {
        return Def(std::string(op) + " fast float " + a + ", " + b);
    }
    std::string Add(const std::string& a, const std::string& b) { return Op("fadd", a, b); }
    std::string Sub(const std::string& a, const std::string& b) { return Op("fsub", a, b); }
    std::string Mul(const std::string& a, const std::string& b) { return Op("fmul", a, b); }
    std::string Cmp(const char* pred, const std::string& a, const std::string& b)
    {
        return Def(std::string("fcmp fast ") + pred + " float " + a + ", " + b);
    }
    std::string Sel(const std::string& c, const std::string& a, const std::string& b)
    {
        return Def("select i1 " + c + ", float " + a + ", float " + b);
    }
    std::string Unary(int op, const std::string& a)
    {
        return Def("call float @dx.op.unary.f32(i32 " + std::to_string(op) + ", float " + a + ")");
    }
    std::string Binary(int op, const std::string& a, const std::string& b)
    {
        return Def("call float @dx.op.binary.f32(i32 " + std::to_string(op) + ", float " + a + ", float " + b + ")");
    }
    std::string Extract(const char* type, const std::string& from, int index)
    {
        return Def(std::string("extractvalue ") + type + " " + from + ", " + std::to_string(index));
    }
};

// The text after `\n  <name> = ` to the end of its line; empty when the function does not define it.
std::string McDefOf(const Module& m, const std::string& name)
{
    const std::string key = "\n  " + name + " = ";
    const size_t p = m.t.find(key);
    if (p == std::string::npos) return std::string();
    const size_t e = m.t.find('\n', p + 1);
    return m.t.substr(p + key.size(), e - (p + key.size()));
}

// True when `v` is `x * 0.5 + 0.5`, the unpacking step of the octahedral normal the game stores in SV_Target1.xy.
bool McIsHalfPlusHalf(const Module& m, const std::string& v)
{
    if (v.empty() || v[0] != '%') return false;
    static const char kHalf[] = ", 5.000000e-01";
    std::string add = McDefOf(m, v);
    ReplaceAll(add, " fast ", " ");
    if (add.rfind("fadd float %", 0) != 0 || add.size() <= sizeof(kHalf) ||
        add.compare(add.size() - (sizeof(kHalf) - 1), sizeof(kHalf) - 1, kHalf) != 0)
        return false;
    const std::string inner = add.substr(11, add.size() - (sizeof(kHalf) - 1) - 11);
    std::string mul = McDefOf(m, inner);
    ReplaceAll(mul, " fast ", " ");
    return mul.rfind("fmul float %", 0) == 0 && mul.size() > sizeof(kHalf) &&
        mul.compare(mul.size() - (sizeof(kHalf) - 1), sizeof(kHalf) - 1, kHalf) == 0;
}

// The output signature's SV_Target<index>: its signature id, float with at least `cols` columns; -1 when absent.
long long McTargetSig(Module& m, int out_list_node, long long index, long long cols_min)
{
    for (const auto& el_ref : Operands(m.Node(out_list_node))) {
        const auto f = Operands(m.Node(NodeRef(el_ref)));
        if (f.size() < 11 || f[1] != "!\"SV_Target\"") continue;
        const auto idx = Operands(m.Node(NodeRef(f[4])));
        long long sig;
        long long comp;
        long long cols;
        long long first;
        if (idx.empty() || !IntOf(idx[0], first) || !IntOf(f[0], sig) || !IntOf(f[2], comp) || !IntOf(f[7], cols))
            continue;
        if (first == index && comp == 9 && cols >= cols_min) return sig;
    }
    return -1;
}

// A storeOutput of one component: the line's span in the text and the value it stores.
struct McStore {
    size_t begin = 0;
    size_t end = 0;
    std::string value;
};

// The one `storeOutput` of (signature id, column); false when there are none or several.
bool McFindStore(const Module& m, long long sig, int col, McStore& s)
{
    const std::string pfx = "call void @dx.op.storeOutput.f32(i32 5, i32 " + std::to_string(sig) + ", i32 0, i8 " +
        std::to_string(col) + ", float ";
    const size_t i = m.t.find(pfx);
    if (i == std::string::npos || m.t.find(pfx, i + 1) != std::string::npos) return false;
    const size_t v0 = i + pfx.size();
    const size_t v1 = m.t.find(')', v0);
    const size_t nl = m.t.find('\n', i);
    if (v1 == std::string::npos || nl == std::string::npos) return false;
    s.begin = m.t.rfind('\n', i) + 1;
    s.end = nl + 1;
    s.value = m.t.substr(v0, v1 - v0);
    return true;
}

bool MatcapText(
    Module& m, const std::string& fn, int sig_node, int res_node, const RfxDxilSpec& spec, RfxDxilResult& out)
{
    std::string& why = out.why;
    out.srv_max_reg = 0;
    if (res_node < 0) {
        why = "entry point has no resources (a pixel shader with no texture is not the body PS)";
        return false;
    }
    if (m.t.find("@dx.op.createHandle(") == std::string::npos) {
        why = "the shader does not use dx.op.createHandle (SM 6.6 createHandleFromBinding form is not handled)";
        return false;
    }
    auto rops = Operands(m.Node(res_node));
    if (rops.size() != 4) {
        why = "resource node has " + std::to_string(rops.size()) + " operands";
        return false;
    }

    // ---- cb0 must be the engine's View buffer ----------------------------------------------------
    long long cb_id = -1;
    if (const int cb_list_node = NodeRef(rops[2]); cb_list_node >= 0)
        for (const auto& rec_ref : Operands(m.Node(cb_list_node))) {
            const auto f = Operands(m.Node(NodeRef(rec_ref)));
            long long id;
            long long space;
            long long lower;
            long long size;
            if (f.size() < 7 || !IntOf(f[0], id) || !IntOf(f[3], space) || !IntOf(f[4], lower) || !IntOf(f[6], size))
                continue;
            if (space == 0 && lower == 0 && f[1].find("View") != std::string::npos && size >= 15 * 16) cb_id = id;
        }
    if (cb_id < 0) {
        why = "cb0 is not the View buffer (no space-0 b0 record of a View type)";
        return false;
    }

    // ---- SRVs: free ids, nothing already declared over the registers we add ------------------------
    const unsigned regs[5] = { spec.matcap_reg, spec.matcap_reg + 1, spec.matcap_reg + 2, spec.matcap_reg + 3,
        spec.matcap_params_reg };
    std::vector<std::string> srv_list;
    const int srv_list_node = NodeRef(rops[0]);
    if (srv_list_node >= 0) srv_list = Operands(m.Node(srv_list_node));
    long long new_id = 0;
    for (const auto& rec_ref : srv_list) {
        const auto f = Operands(m.Node(NodeRef(rec_ref)));
        long long id;
        long long space;
        long long lower;
        long long range;
        if (f.size() < 7 || !IntOf(f[0], id) || !IntOf(f[3], space) || !IntOf(f[4], lower) || !IntOf(f[5], range)) {
            why = "SRV record " + rec_ref + " did not parse";
            return false;
        }
        if (id >= new_id) new_id = id + 1;
        const bool unbounded = (range < 0 || range >= 0xFFFFFFFFll);
        if (space != 0 || unbounded) {
            why = "the shader reads an SRV outside the fixed space-0 registers (space " + std::to_string(space) +
                ", t" + std::to_string(lower) + (unbounded ? ", unbounded" : "") +
                "): a one-table shadow bind would leave it dangling";
            return false;
        }
        for (unsigned reg : regs)
            if (lower <= (long long)reg && lower + range > (long long)reg) {
                why = "the shader already declares t" + std::to_string(lower) + " x" + std::to_string(range) +
                    " over t" + std::to_string(reg);
                return false;
            }
        if ((unsigned)(lower + range - 1) > out.srv_max_reg) out.srv_max_reg = (unsigned)(lower + range - 1);
    }

    // ---- a plain sampler, as the ini's s0 -----------------------------------------------------------
    long long smp_id = -1;
    long long smp_reg = 0;
    if (const int smp_list_node = NodeRef(rops[3]); smp_list_node >= 0)
        for (const auto& rec_ref : Operands(m.Node(smp_list_node))) {
            const auto f = Operands(m.Node(NodeRef(rec_ref)));
            long long id;
            long long space;
            long long lower;
            long long range;
            long long kind;
            if (f.size() < 7 || !IntOf(f[0], id) || !IntOf(f[3], space) || !IntOf(f[4], lower) || !IntOf(f[5], range) ||
                !IntOf(f[6], kind))
                continue;
            if (kind != 0 || range == 0) continue;
            if (space == 0 && lower == 0) {
                smp_id = id;
                smp_reg = 0;
                break;
            }
            if (smp_id < 0) {
                smp_id = id;
                smp_reg = lower;
            }
        }
    if (smp_id < 0) {
        why = "no plain sampler";
        return false;
    }

    // ---- signatures: TEXCOORD0 (the mask's uv), SV_Target0 / 1 / 3 ----------------------------------
    const auto sops = Operands(m.Node(sig_node));
    const int in_list_node = sops.empty() ? -1 : NodeRef(sops[0]);
    const int out_list_node = sops.size() > 1 ? NodeRef(sops[1]) : -1;
    if (in_list_node < 0 || out_list_node < 0) {
        why = "no input or output signature";
        return false;
    }
    long long uv_sig = -1;
    int uv_elem_node = -1;
    std::vector<std::string> uv_elem;
    for (const auto& el_ref : Operands(m.Node(in_list_node))) {
        const auto f = Operands(m.Node(NodeRef(el_ref)));
        if (f.size() < 11 || f[1] != "!\"TEXCOORD\"") continue;
        const auto idx = Operands(m.Node(NodeRef(f[4])));
        long long sig;
        long long comp;
        long long cols;
        long long first;
        if (idx.empty() || !IntOf(idx[0], first) || first != 0) continue;
        if (!IntOf(f[0], sig) || !IntOf(f[2], comp) || !IntOf(f[7], cols)) continue;
        if (comp != 9 || cols < 2) {
            why = "TEXCOORD0 is not a float2+ (type " + std::to_string(comp) + ", cols " + std::to_string(cols) + ")";
            return false;
        }
        uv_sig = sig;
        uv_elem_node = NodeRef(el_ref);
        uv_elem = f;
        break;
    }
    if (uv_sig < 0) {
        why = "the input signature has no TEXCOORD0";
        return false;
    }
    const long long o0 = McTargetSig(m, out_list_node, 0, 3);
    const long long o1 = McTargetSig(m, out_list_node, 1, 2);
    const long long o3 = McTargetSig(m, out_list_node, 3, 3);
    if (o0 < 0 || o1 < 0 || o3 < 0) {
        why = "no float SV_Target0 + SV_Target1 + SV_Target3";
        return false;
    }

    // ---- the stores: each of the eight components stored once, all in one straight run of the last block ------
    McStore st0[3];
    McStore st3[3];
    McStore st1[2];
    size_t lo = std::string::npos;
    size_t hi = 0;
    auto note = [&](const McStore& s) {
        lo = (std::min)(lo, s.begin);
        hi = (std::max)(hi, s.end);
    };
    for (int c = 0; c < 3; ++c) {
        if (!McFindStore(m, o0, c, st0[c]) || !McFindStore(m, o3, c, st3[c])) {
            why = "SV_Target0 / SV_Target3 column " + std::to_string(c) + " is not stored exactly once";
            return false;
        }
        note(st0[c]);
        note(st3[c]);
    }
    for (int c = 0; c < 2; ++c) {
        if (!McFindStore(m, o1, c, st1[c])) {
            why = "SV_Target1 column " + std::to_string(c) + " is not stored exactly once";
            return false;
        }
        note(st1[c]);
        if (!McIsHalfPlusHalf(m, st1[c].value)) {
            why = "SV_Target1." + std::string(c ? "y" : "x") + " is not the octahedral 0.5 * oct + 0.5 (" +
                st1[c].value + " = " + McDefOf(m, st1[c].value) + ")";
            return false;
        }
    }
    {
        const std::string run = m.t.substr(lo, hi - lo);
        if (run.find("\n; <label>") != std::string::npos || run.find(" br ") != std::string::npos ||
            run.find("ret void") != std::string::npos) {
            why = "the output stores are not one straight run (a branch or label lies between them)";
            return false;
        }
    }

    // ---- TEXCOORD0's used mask must cover x and y. The tag node may be shared: never edit it. ----------------
    {
        std::vector<std::string> tags;
        const int tag_node = NodeRef(uv_elem[10]);
        if (tag_node >= 0) tags = Operands(m.Node(tag_node));
        bool have = false;
        bool changed = false;
        for (size_t i = 0; i + 1 < tags.size(); i += 2) {
            long long tag;
            long long mask;
            if (IntOf(tags[i], tag) && tag == 3 && IntOf(tags[i + 1], mask)) {
                have = true;
                if ((mask & 3) != 3) {
                    tags[i + 1] = "i32 " + std::to_string(mask | 3);
                    changed = true;
                }
            }
        }
        if (!have) {
            tags.push_back("i32 3");
            tags.push_back("i32 3");
            changed = true;
        }
        if (changed) {
            uv_elem[10] = "!" + std::to_string(m.AddNode(Join(tags)));
            m.SetNode(uv_elem_node, "!" + std::to_string(uv_elem_node) + " = " + Join(uv_elem));
        }
    }

    // ---- the new SRV records: four Texture2D (layers, masks) and the params Buffer<float4> -------------------
    long long srv_ids[5];
    {
        const int tag = m.AddNode("!{i32 0, i32 9}"); // element type: float
        for (int k = 0; k < 5; ++k) {
            srv_ids[k] = new_id + k;
            const int rec = m.AddNode("!{i32 " + std::to_string(srv_ids[k]) + ", " + (k < 4 ? kTexType : kBufType) +
                "* undef, !\"\", i32 0, i32 " + std::to_string(regs[k]) + ", i32 1, i32 " + (k < 4 ? "2" : "10") +
                ", i32 0, !" + std::to_string(tag) + "}");
            srv_list.push_back("!" + std::to_string(rec));
        }
        if (srv_list_node >= 0)
            m.SetNode(srv_list_node, "!" + std::to_string(srv_list_node) + " = " + Join(srv_list));
        else {
            rops[0] = "!" + std::to_string(m.AddNode(Join(srv_list)));
            m.SetNode(res_node, "!" + std::to_string(res_node) + " = " + Join(rops));
        }
    }

    // ---- types and declarations the injection needs, placed before the function ------------------------------
    std::string pre;
    auto need_type = [&](const std::string& name, const std::string& body) {
        if (m.t.find("\n" + name + " = type") == std::string::npos) pre += name + " = type " + body + "\n";
    };
    need_type("%dx.types.Handle", "{ i8* }");
    need_type("%dx.types.ResRet.f32", "{ float, float, float, float, i32 }");
    need_type("%dx.types.CBufRet.f32", "{ float, float, float, float }");
    need_type("%dx.types.Dimensions", "{ i32, i32, i32, i32 }");
    need_type(kTexType, std::string("{ <4 x float>, ") + kMipsType + " }");
    need_type(kMipsType, "{ i32 }");
    need_type(kBufType, "{ <4 x float> }");
    auto need_decl = [&](const char* name, const char* decl) {
        if (m.t.find(std::string("@dx.op.") + name + "(i32,") == std::string::npos) pre += std::string(decl) + "\n";
    };
    need_decl("loadInput.f32", "declare float @dx.op.loadInput.f32(i32, i32, i32, i8, i32) nounwind readnone");
    need_decl("getDimensions",
        "declare %dx.types.Dimensions @dx.op.getDimensions(i32, %dx.types.Handle, i32) nounwind readonly");
    need_decl("sample.f32",
        "declare %dx.types.ResRet.f32 @dx.op.sample.f32(i32, %dx.types.Handle, %dx.types.Handle, float, float, float, float, i32, i32, i32, float) nounwind readonly");
    need_decl("bufferLoad.f32",
        "declare %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32, %dx.types.Handle, i32, i32) nounwind readonly");
    need_decl("cbufferLoadLegacy.f32",
        "declare %dx.types.CBufRet.f32 @dx.op.cbufferLoadLegacy.f32(i32, %dx.types.Handle, i32) nounwind readonly");
    need_decl("unary.f32", "declare float @dx.op.unary.f32(i32, float) nounwind readnone");
    need_decl("binary.f32", "declare float @dx.op.binary.f32(i32, float, float) nounwind readnone");
    need_decl("dot3.f32", "declare float @dx.op.dot3.f32(i32, float, float, float, float, float, float) nounwind readnone");

    // ---- the blend ------------------------------------------------------------------------------------------
    McIr ir;
    auto lit = [](float f) { return FloatLit(f); };
    auto handle = [&](const char* name, int cls, long long id, long long reg) {
        ir.text += std::string("  %mc_") + name + " = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 " +
            std::to_string(cls) + ", i32 " + std::to_string(id) + ", i32 " + std::to_string(reg) + ", i1 false)\n";
        return std::string("%mc_") + name;
    };
    const std::string tex_h[2] = { handle("t1", 0, srv_ids[0], regs[0]), handle("t2", 0, srv_ids[1], regs[1]) };
    const std::string mask_h[2] = { handle("m1", 0, srv_ids[2], regs[2]), handle("m2", 0, srv_ids[3], regs[3]) };
    const std::string prm_h = handle("pr", 0, srv_ids[4], regs[4]);
    const std::string smp_h = handle("sm", 3, smp_id, smp_reg);
    const std::string cb_h = handle("cb", 2, cb_id, 0);
    const std::string kRes = "%dx.types.ResRet.f32";
    const std::string kCb = "%dx.types.CBufRet.f32";

    // The ini's parameters.
    std::string par[2][4];
    for (int t = 0; t < 2; ++t) {
        const std::string ld = ir.Def("call " + kRes + " @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle " + prm_h +
            ", i32 " + std::to_string(t) + ", i32 undef)");
        for (int c = 0; c < 4; ++c) par[t][c] = ir.Extract(kRes.c_str(), ld, c);
    }

    // The world normal, decoded from the octahedral value the shader stores, as a view-space xy.
    const std::string fx = ir.Add(ir.Mul(st1[0].value, lit(2.f)), lit(-1.f));
    const std::string fy = ir.Add(ir.Mul(st1[1].value, lit(2.f)), lit(-1.f));
    const std::string nz = ir.Sub(ir.Sub(lit(1.f), ir.Unary(6, fx)), ir.Unary(6, fy));
    const std::string fold = ir.Binary(35, ir.Sub(lit(0.f), nz), lit(0.f));
    const std::string unfold = ir.Sub(lit(0.f), fold);
    const std::string nx = ir.Add(fx, ir.Sel(ir.Cmp("oge", fx, lit(0.f)), unfold, fold));
    const std::string ny = ir.Add(fy, ir.Sel(ir.Cmp("oge", fy, lit(0.f)), unfold, fold));
    const std::string len2 = ir.Def("call float @dx.op.dot3.f32(i32 55, float " + nx + ", float " + ny + ", float " +
        nz + ", float " + nx + ", float " + ny + ", float " + nz + ")");
    const std::string inv = ir.Unary(25, len2);
    const std::string wn[3] = { ir.Mul(nx, inv), ir.Mul(ny, inv), ir.Mul(nz, inv) };
    std::string view_xy[2];
    for (int c = 0; c < 2; ++c) {
        std::string sum;
        for (int row = 0; row < 3; ++row) {
            const std::string r = ir.Def("call " + kCb + " @dx.op.cbufferLoadLegacy.f32(i32 59, %dx.types.Handle " +
                cb_h + ", i32 " + std::to_string(12 + row) + ")");
            const std::string term = ir.Mul(wn[row], ir.Extract(kCb.c_str(), r, c));
            sum = sum.empty() ? term : ir.Add(sum, term);
        }
        view_xy[c] = sum;
    }
    const std::string mu = ir.Unary(7, ir.Add(ir.Mul(view_xy[0], lit(0.5f)), lit(0.5f)));
    const std::string mv = ir.Unary(7, ir.Add(ir.Mul(ir.Sub(lit(0.f), view_xy[1]), lit(0.5f)), lit(0.5f)));

    const std::string uv_u = ir.Def("call float @dx.op.loadInput.f32(i32 4, i32 " + std::to_string(uv_sig) +
        ", i32 0, i8 0, i32 undef)");
    const std::string uv_v = ir.Def("call float @dx.op.loadInput.f32(i32 4, i32 " + std::to_string(uv_sig) +
        ", i32 0, i8 1, i32 undef)");
    auto sample = [&](const std::string& tex, const std::string& u, const std::string& v) {
        return ir.Def("call " + kRes + " @dx.op.sample.f32(i32 60, %dx.types.Handle " + tex +
            ", %dx.types.Handle " + smp_h + ", float " + u + ", float " + v +
            ", float undef, float undef, i32 0, i32 0, i32 undef, float undef)");
    };
    auto bound = [&](const std::string& tex) { // a bound texture has a size
        const std::string d = ir.Def("call %dx.types.Dimensions @dx.op.getDimensions(i32 72, %dx.types.Handle " +
            tex + ", i32 0)");
        const std::string w = ir.Extract("%dx.types.Dimensions", d, 0);
        const std::string h = ir.Extract("%dx.types.Dimensions", d, 1);
        return ir.Def("icmp ne i32 " + ir.Def("add i32 " + w + ", " + h) + ", 0");
    };

    // The scene colour ("emissive") and the base colour, as the shader stores them.
    std::string emi[3] = { st0[0].value, st0[1].value, st0[2].value };
    std::string base[3] = { st3[0].value, st3[1].value, st3[2].value };
    for (int layer = 0; layer < 2; ++layer) {
        const std::string blend = par[0][layer * 2];
        const std::string opacity = par[0][layer * 2 + 1];
        const std::string strength = par[1][layer];
        const std::string mask_opacity = par[1][2 + layer];
        const std::string tex_on = bound(tex_h[layer]);
        const std::string mask_on = bound(mask_h[layer]);
        // factor = opacity, times 1 + mask_opacity * (mask - 1) when a mask is bound
        const std::string mask_x = ir.Extract(kRes.c_str(), sample(mask_h[layer], uv_u, uv_v), 0);
        const std::string masked =
            ir.Mul(opacity, ir.Add(ir.Mul(mask_opacity, ir.Add(mask_x, lit(-1.f))), lit(1.f)));
        const std::string factor = ir.Sel(mask_on, masked, opacity);
        const std::string go = ir.Def("and i1 " + tex_on + ", " + ir.Cmp("ogt", factor, lit(0.0001f)));
        const std::string gain = ir.Sel(ir.Cmp("oeq", strength, lit(0.f)), lit(1.f), strength);
        const std::string smp = sample(tex_h[layer], mu, mv);
        const std::string mode_f = ir.Binary(36, ir.Binary(35, blend, lit(0.f)), lit(255.f));
        const std::string mode = ir.Def("fptoui float " + mode_f + " to i32");
        const std::string is0 = ir.Def("icmp eq i32 " + mode + ", 0");
        const std::string is1 = ir.Def("icmp eq i32 " + mode + ", 1");
        const std::string is2 = ir.Def("icmp eq i32 " + mode + ", 2");
        const std::string is4 = ir.Def("icmp eq i32 " + mode + ", 4");
        const std::string on_emissive = ir.Def("icmp ule i32 " + mode + ", 2");
        for (int c = 0; c < 3; ++c) {
            const std::string col = ir.Mul(ir.Extract(kRes.c_str(), smp, c), gain);
            const std::string scaled = ir.Mul(col, factor);
            const std::string e_add = ir.Add(emi[c], scaled);
            const std::string e_lerp = ir.Add(emi[c], ir.Mul(factor, ir.Sub(col, emi[c])));
            const std::string e_screen = ir.Add(emi[c], ir.Mul(scaled, ir.Sub(lit(1.f), emi[c])));
            const std::string e_new = ir.Sel(is1, e_add, ir.Sel(is0, e_lerp, ir.Sel(is2, e_screen, emi[c])));
            const std::string b_lerp = ir.Add(base[c], ir.Mul(factor, ir.Sub(col, base[c])));
            const std::string b_mul = ir.Mul(base[c], ir.Add(ir.Mul(factor, ir.Sub(col, lit(1.f))), lit(1.f)));
            const std::string b_new = ir.Sel(is4, b_lerp, ir.Sel(on_emissive, base[c], b_mul));
            emi[c] = ir.Sel(go, e_new, emi[c]);
            base[c] = ir.Sel(go, b_new, base[c]);
        }
    }
    std::string stores;
    for (int c = 0; c < 3; ++c) {
        stores += "  call void @dx.op.storeOutput.f32(i32 5, i32 " + std::to_string(o0) + ", i32 0, i8 " +
            std::to_string(c) + ", float " + emi[c] + ")\n";
        stores += "  call void @dx.op.storeOutput.f32(i32 5, i32 " + std::to_string(o3) + ", i32 0, i8 " +
            std::to_string(c) + ", float " + base[c] + ")\n";
    }

    // ---- the edit: the six old stores leave the run, the blend and the new stores follow it -------------------
    std::vector<std::pair<size_t, size_t>> gone;
    for (int c = 0; c < 3; ++c) {
        gone.emplace_back(st0[c].begin, st0[c].end);
        gone.emplace_back(st3[c].begin, st3[c].end);
    }
    std::sort(gone.begin(), gone.end());
    std::string run;
    size_t at = lo;
    for (const auto& g : gone) {
        run += m.t.substr(at, g.first - at);
        at = g.second;
    }
    run += m.t.substr(at, hi - at);
    m.t.replace(lo, hi - lo, run + ir.text + stores);

    const size_t def = m.t.find("\ndefine void " + fn + "(");
    if (def == std::string::npos) {
        why = "no `define void " + fn + "(`";
        return false;
    }
    m.t.insert(def + 1, pre);
    why = "matcap: o0 = output " + std::to_string(o0) + ", o1 (octahedral normal) = output " + std::to_string(o1) +
        ", o3 = output " + std::to_string(o3) + ", cb" + "0 = View (id " + std::to_string(cb_id) + "), SRV ids " +
        std::to_string(srv_ids[0]) + ".." + std::to_string(srv_ids[4]) + " at t" + std::to_string(regs[0]) + "..t" +
        std::to_string(regs[3]) + " / t" + std::to_string(regs[4]) + ", sampler s" + std::to_string(smp_reg) +
        ", its own SRVs reach t" + std::to_string(out.srv_max_reg);
    return true;
}
