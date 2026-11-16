// matcap_patch.cpp -- OURS. See matcap_patch.h.
#include "matcap_patch.h"
#include "../addon_common.h"
#include "../addon/addon.h"
#include "../core/fnv.h"
#include "../core/log.h"
#include "../hooks/cmd_bind_hooks.h"
#include "../hooks/command_list_registry.h"
#include "../hooks/pipeline_registry.h"
#include "../instrumentation/diag_gate.h"
#include "../migoto/dx12_pipeline.h"
#include "../migoto/dx12_wrap.h"
#include "../migoto/port_api.h"
#include "../mods/shader_alias.h"
#include "../mods/wwmi_runtime.h"
#include "../rabbitfx/rfx_dxil.h"
#include "../rabbitfx/rfx_patched_ps.h"
#include "../rabbitfx/rfx_patch_srvs.h"
#include "../build_stamp.h" // WWMI_RFX_SRC_HASH: a cached patch is reused only by the patcher that wrote it
#include <cstring>
#include <string>

using namespace reshade::api;

MatcapWorker g_matcap;

namespace {

struct PatchedPs {
    std::vector<uint8_t> code; // empty: the patch did not apply, said once and never retried
    unsigned max_reg = 0;
    bool cached = false;
};

struct CacheHdr {
    uint32_t magic;
    uint32_t version;
    uint64_t src_hash;
    uint32_t src_size;
    uint32_t code_size;
    uint32_t max_reg;
    uint32_t reserved;
};
constexpr uint32_t kCacheMagic = 0x5043414D; // "MACP"
constexpr uint64_t kCacheSalt = 0x4D41544341504658ull; // keeps a matcap file apart from the RabbitFX patches of the shader

/** The cache file of the patch of the shader whose bytes hash to `src_hash`. */
std::wstring CachePath(uint64_t src_hash)
{
    wchar_t leaf[40];
    swprintf_s(leaf, L"\\%016llx.mcp", (unsigned long long)(src_hash ^ kCacheSalt));
    return g_addon.shader_cache_dir + leaf;
}

/**
 * Reads a patch from the disk cache.
 * @param src_hash hash of the shader bytes the patch was made from
 * @param src_size their size
 * @param out receives the patch (empty code for a shader the patch did not apply to)
 * @return true for a file this build wrote for this shader
 */
bool CacheLoad(uint64_t src_hash, size_t src_size, PatchedPs& out)
{
    if (g_addon.shader_cache_dir.empty()) return false;
    const std::vector<uint8_t> f = WwmiReadFile(CachePath(src_hash).c_str());
    CacheHdr h;
    if (f.size() < sizeof h) return false;
    memcpy(&h, f.data(), sizeof h);
    if (h.magic != kCacheMagic || h.version != WWMI_RFX_SRC_HASH || h.src_hash != src_hash ||
        h.src_size != (uint32_t)src_size || f.size() != sizeof h + (size_t)h.code_size)
        return false;
    out.code.assign(f.begin() + sizeof h, f.end());
    out.max_reg = h.max_reg;
    return true;
}

/** Writes a patch to the disk cache; a failed write leaves no file. */
void CacheStore(uint64_t src_hash, size_t src_size, const PatchedPs& p)
{
    if (g_addon.shader_cache_dir.empty()) return;
    CreateDirectoryW(g_addon.shader_cache_dir.c_str(), nullptr);
    const std::wstring path = CachePath(src_hash);
    const std::wstring tmp = path + L".tmp";
    const HANDLE f = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return;
    const CacheHdr h{ kCacheMagic, WWMI_RFX_SRC_HASH, src_hash, (uint32_t)src_size, (uint32_t)p.code.size(), p.max_reg, 0 };
    DWORD wr = 0;
    bool ok = WriteFile(f, &h, sizeof h, &wr, nullptr) && wr == sizeof h;
    if (ok && !p.code.empty())
        ok = WriteFile(f, p.code.data(), (DWORD)p.code.size(), &wr, nullptr) && wr == p.code.size();
    CloseHandle(f);
    if (!ok || !MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) DeleteFileW(tmp.c_str());
}

// The worker thread's patched shaders, by the hash of the bytes they were made from. WORKER THREAD ONLY.
std::unordered_map<uint64_t, std::shared_ptr<PatchedPs>> g_patched;

} // namespace

void MatcapWorker::Build(Job& job)
{
    reshade::api::device* const dev = job.dev;
    const uint64_t src_pso = job.src_pso;
    const uint64_t key = job.key;
    unsigned max_reg = 0;
    auto settle = [&](ID3D12PipelineState* v) {
        bool orphan = false;
        {
            std::lock_guard l(mutex_);
            if (!pending_.erase(key))
                orphan = true;
            else
                variants_[key] = Made{ v, max_reg };
        }
        if (orphan && v) {
            logger::write("MATCAP variant src=%llx: the source PSO was destroyed while its patch was in flight -- variant released",
                src_pso);
            dev->destroy_pipeline(reshade::api::pipeline{ reinterpret_cast<uint64_t>(v) });
        }
    };
    std::vector<pipeline_subobject>& subs = job.clone.subs;
    // A GBuffer pass only: the patch reads and writes SV_Target0 / 1 / 3.
    for (const auto& so : subs)
        if (so.type == pipeline_subobject_type::render_target_formats) {
            const auto* fm = static_cast<const format*>(so.data);
            if (so.count < 4 || fm[3] == format::unknown) return settle(nullptr);
            break;
        }
    pipeline_subobject* ps_so = nullptr;
    for (auto& so : subs)
        if (so.type == pipeline_subobject_type::pixel_shader && so.count >= 1 && so.data) {
            ps_so = &so;
            break;
        }
    if (!ps_so) return settle(nullptr);
    const shader_desc src_ps = *static_cast<const shader_desc*>(ps_so->data);
    if (!src_ps.code || !src_ps.code_size) return settle(nullptr);
    const void* const in_code = job.base_code ? static_cast<const void*>(job.base_code->data()) : src_ps.code;
    const size_t in_size = job.base_code ? job.base_code->size() : (size_t)src_ps.code_size;
    const uint64_t in_hash = Fnv1a64(in_code, in_size);

    std::shared_ptr<PatchedPs> patched;
    if (auto it = g_patched.find(in_hash); it != g_patched.end())
        patched = it->second;
    else {
        patched = std::make_shared<PatchedPs>();
        patched->cached = CacheLoad(in_hash, in_size, *patched);
        if (patched->cached)
            logger::trace("MATCAP-CACHE ps=%016llx: %zu bytes from the disk cache", (unsigned long long)in_hash,
                patched->code.size());
        else {
            std::string why;
            const ULONGLONG t0 = GetTickCount64();
            RfxDxilResult r;
            const bool init_ok = RfxDxilInit((g_addon.cso_dir + L"\\dxc").c_str(), why);
            RfxDxilSpec spec;
            spec.matcap = true;
            spec.matcap_reg = kMatcapReg;
            spec.matcap_params_reg = kMatcapParamsReg;
            const bool ok = init_ok && RfxDxilPatch(in_code, in_size, spec, r);
            if (ok) {
                patched->code = std::move(r.code);
                patched->max_reg = r.srv_max_reg;
            } else if (!r.why.empty())
                why = r.why;
            logger::write("MATCAP ps=%016llx (%zu bytes): %s in %llu ms -- %s", (unsigned long long)in_hash, in_size,
                ok ? "PATCHED" : "NOT patched", (unsigned long long)(GetTickCount64() - t0),
                ok ? r.why.c_str() : why.c_str());
            if (r.settled) CacheStore(in_hash, in_size, *patched);
        }
        g_patched.emplace(in_hash, patched);
    }
    if (patched->code.empty()) return settle(nullptr);
    max_reg = patched->max_reg;

    shader_desc ps = src_ps;
    ps.code = patched->code.data();
    ps.code_size = patched->code.size();
    ps_so->data = &ps;
    ps_so->count = 1;
    reshade::api::pipeline out = { 0 };
    const bool ok = dev->create_pipeline(job.clone.layout, (uint32_t)subs.size(), subs.data(), &out);
    ID3D12PipelineState* variant = ok ? reinterpret_cast<ID3D12PipelineState*>(out.handle) : nullptr;
    logger::write("MATCAP variant src=%llx ps=%016llx -> %s", src_pso, (unsigned long long)in_hash,
        variant ? "OK" : "FAILED(create)");
    if (!variant && patched->cached) {
        DeleteFileW(CachePath(in_hash).c_str());
        logger::write("MATCAP-CACHE ps=%016llx: its cached patch built no pipeline -- the cache file is removed, patched afresh next launch",
            (unsigned long long)in_hash);
    }
    return settle(variant);
}

void MatcapWorker::Run()
{
    for (;;) {
        Job job;
        {
            std::unique_lock l(mutex_);
            cv_.wait(l, [this] { return stop_.load(std::memory_order_acquire) || !queue_.empty(); });
            if (stop_.load(std::memory_order_acquire)) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        Build(job);
    }
}

void MatcapWorker::Stop()
{
    {
        std::lock_guard l(mutex_);
        stop_.store(true, std::memory_order_release);
        queue_.clear();
        pending_.clear();
    } // an in-flight build then settles as an orphan
    cv_.notify_all();
    std::lock_guard t(thread_mutex_);
    if (worker_.t.joinable()) {
        worker_.t.join();
        logger::write("MATCAP worker stopped (device destroyed / addon unloading)");
    }
}

void MatcapWorker::Wake()
{
    {
        std::lock_guard t(thread_mutex_);
        if (!worker_.t.joinable() && !stop_.load(std::memory_order_acquire)) {
            worker_.t = std::thread([this] { Run(); });
            logger::write("MATCAP worker started");
        }
    }
    cv_.notify_one();
}

ID3D12PipelineState* MatcapWorker::Variant(
    reshade::api::device* dev, uint64_t src_pso, MatcapBase kind, CodeBlob base_code, unsigned* ps_max_reg)
{
    if (ps_max_reg) *ps_max_reg = 0;
    if (!dev || !src_pso) return nullptr;
    if (kind != MatcapBase::Game && !base_code) return nullptr;
    const uint64_t key = src_pso | ((uint64_t)kind << 60);
    {
        std::lock_guard l(mutex_);
        auto it = variants_.find(key);
        if (it != variants_.end()) {
            if (ps_max_reg) *ps_max_reg = it->second.max_reg;
            return it->second.pso;
        }
        if (pending_.count(key)) return nullptr;
    } // in flight: this draw goes without the matcap

    Job job{ dev, src_pso, key, kind == MatcapBase::Game ? CodeBlob() : base_code, {} };
    if (!g_pso_clones.DeepCopy(src_pso, job.clone)) {
        std::lock_guard lv(mutex_);
        if (miss_logged_.insert(src_pso).second)
            logger::write("MATCAP variant src=%llx -> none: no clone of this PSO (created before the addon)", src_pso);
        return nullptr;
    }
    {
        std::lock_guard l(mutex_);
        if (stop_.load(std::memory_order_acquire)) return nullptr;
        if (!pending_.insert(key).second) return nullptr; // another draw thread queued it first
        queue_.push_back(std::move(job));
        logger::trace("MATCAP queued src=%llx base=%u (drawn without it until built; %zu queued)", src_pso,
            (unsigned)kind, queue_.size());
    }
    Wake();
    return nullptr;
}

void MatcapWorker::ForgetSource(uint64_t src_pso, std::vector<ID3D12PipelineState*>& dead)
{
    std::lock_guard l(mutex_);
    miss_logged_.erase(src_pso);
    constexpr uint64_t kSrcMask = ~(7ull << 60);
    for (unsigned d = 0; d < 4; ++d) {
        const uint64_t key = src_pso | ((uint64_t)d << 60);
        if (auto it = variants_.find(key); it != variants_.end()) {
            if (it->second.pso) dead.push_back(it->second.pso);
            variants_.erase(it);
        }
        pending_.erase(key);
    }
    for (auto it = queue_.begin(); it != queue_.end();)
        if ((it->key & kSrcMask) == src_pso)
            it = queue_.erase(it);
        else
            ++it;
}

MatcapPlan MatcapPlanDraw(reshade::api::command_list* cmd_list, uint64_t src_pso, bool rfx_variant_in_use,
    bool decensor_on, uint32_t srv_gen0, int mod, int comp)
{
    MatcapPlan plan;
    if (!cmd_list || !src_pso || !PortApi::ShaderRegexGroupLoaded(kMatcapNamespace, kMatcapRegexGroup)) return plan;

    // The textures the mod bound in this section; a slot it never set is null, and so is one it cleared.
    const MigotoPort::PipeState& st = MigotoPort::State();
    const int si = MigotoPort::StageIndex('p');
    auto view = [&](unsigned reg) {
        if ((int32_t)(st.srv_gen[si][reg] - srv_gen0) <= 0) return D3D12_CPU_DESCRIPTOR_HANDLE{ 0 };
        MigotoPort::WrapView* wv = st.srv[si][reg] ? MigotoPort::AsWrapView(st.srv[si][reg]) : nullptr;
        return (wv && wv->kind == 't') ? wv->cpu : D3D12_CPU_DESCRIPTOR_HANDLE{ 0 };
    };
    const D3D12_CPU_DESCRIPTOR_HANDLE layer1 = view(kMatcapLayer1Reg);
    const D3D12_CPU_DESCRIPTOR_HANDLE layer2 = view(kMatcapLayer2Reg);
    if (!layer1.ptr && !layer2.ptr) return plan;
    const D3D12_CPU_DESCRIPTOR_HANDLE mask1 = view(kMatcapMask1Reg);
    const D3D12_CPU_DESCRIPTOR_HANDLE mask2 = view(kMatcapMask2Reg);

    // The shader the matcap goes over: what this draw would run without it.
    MatcapBase kind = MatcapBase::Game;
    CodeBlob base_code;
    if (rfx_variant_in_use) {
        kind = decensor_on ? MatcapBase::RfxDecensored : MatcapBase::Rfx;
        base_code = g_patched_ps.Rfx(src_pso, decensor_on);
    } else if (decensor_on) {
        base_code = g_patched_ps.Decensored(src_pso);
        if (base_code) kind = MatcapBase::Decensored;
    }
    if (kind != MatcapBase::Game && !base_code) return plan;

    unsigned ps_max = 0;
    ID3D12PipelineState* const variant = g_matcap.Variant(cmd_list->get_device(), src_pso, kind, base_code, &ps_max);
    if (!variant) return plan;
    const unsigned top_reg = ps_max > kMatcapParamsReg ? ps_max : kMatcapParamsReg;
    unsigned win_base = 0;
    unsigned win_count = 0;
    if (!g_rfx_bind.Param0Covers(cmd_list, top_reg, &win_base, &win_count)) {
        if (DiagOnce("MATCAP-WINDOW", (int)(((src_pso ^ (src_pso >> 32)) * 31u) & 0x7fffffff)))
            logger::write(
                "MATCAP mod=%d comp=%d pso=%llx: the PS reads up to t%u and the patch needs t%u, the draw's first SRV table is t%u x%u -- drawn without it",
                mod, comp, (unsigned long long)src_pso, ps_max, top_reg, win_base, win_count);
        return plan;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE null_tex{ 0 };
    if (!g_rfx_patch_srvs.NullSrvs(&null_tex, nullptr)) return plan;
    float params[8];
    PortApi::GetIniParam(kMatcapParams1, params);
    PortApi::GetIniParam(kMatcapParams2, params + 4);
    const D3D12_CPU_DESCRIPTOR_HANDLE prm = g_rfx_patch_srvs.ParamsSrvN(params, 2);
    if (!prm.ptr) return plan;

    plan.binds.push_back({ (int)kMatcapReg, layer1.ptr ? layer1 : null_tex, false });
    plan.binds.push_back({ (int)kMatcapReg + 1, layer2.ptr ? layer2 : null_tex, false });
    plan.binds.push_back({ (int)kMatcapReg + 2, mask1.ptr ? mask1 : null_tex, false });
    plan.binds.push_back({ (int)kMatcapReg + 3, mask2.ptr ? mask2 : null_tex, false });
    plan.binds.push_back({ (int)kMatcapParamsReg, prm, false });
    plan.pso = variant;
    plan.ps_max_reg = ps_max;
    plan.top_reg = top_reg;

    // The group's `post run` (the ini's Clear) follows the shaders the group patches: link it to this one, once.
    uint64_t ps_hash = 0;
    cmd_list->get_private_data(s_iid_ps_hash, &ps_hash);
    const uint64_t alias = AliasShaderHash(ps_hash);
    if (alias && !PortApi::ShaderRegexLinked(alias, kMatcapNamespace, kMatcapRegexGroup))
        PortApi::LinkShaderRegexGroup(alias, kMatcapNamespace, kMatcapRegexGroup);
    if (DiagOnce("MATCAP-DRAW", CompKey(mod, comp) * 16 + (int)kind * 4 + (layer1.ptr ? 1 : 0) + (layer2.ptr ? 2 : 0)))
        logger::write(
            "MATCAP draw mod=%d comp=%d pso=%llx base=%u -> variant %p; layer1 %s (mask %s), layer2 %s (mask %s); x125..w125 = %g %g %g %g, x126..w126 = %g %g %g %g",
            mod, comp, (unsigned long long)src_pso, (unsigned)kind, (void*)variant, layer1.ptr ? "bound" : "none",
            mask1.ptr ? "bound" : "none", layer2.ptr ? "bound" : "none", mask2.ptr ? "bound" : "none", params[0],
            params[1], params[2], params[3], params[4], params[5], params[6], params[7]);
    return plan;
}
