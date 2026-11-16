// rt/rt_portable.cpp -- the portable build's own RT module (WwmiRT=portable), used in place of rt_stub.cpp.
//
// Upstream's RT module is not public, so this one is written from its headers. It records the vertex buffers the
// game builds skinned BLAS from (the skin-cache output "rings") and names the mod character a ring belongs to by its
// index buffer. PTSKIN (below) then skins a mod's own vertices for the game's ray-tracing draw path. The BLAS itself
// is still built from the game's own mesh, so ray-traced shadows and reflections show the game's model.
#include <imgui.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <wrl/client.h>
#include <string_view>
#include <string>
#include <atomic>
#include "../addon_common.h"
#include "../core/frame_clock.h"
#include "../core/log.h"
#include "../addon/home_dir.h"
#include "../mods/character_recognition.h"
#include "../mods/layout_registry.h"
#include "../hooks/descriptor_shadow.h"
#include "../hooks/resource_tracker.h"
#include "../instrumentation/removal_pin.h"
#include "../migoto/dx12_bridge.h" // MigotoPort::kResStateUnknown
#include "../hooks/command_list_registry.h"
#include "../hooks/pipeline_registry.h"    // g_pipelines.SrvRegs
#include "../rabbitfx/rabbitfx_bind.h"     // g_rfx_bind, ShadowSlotOvr
#include "../mods/game_cb_reader.h"
#include "../mods/wwmi_runtime.h" // WwmiReadFile, WwmiMakeBuffer
#include "../addon/addon.h"        // g_addon.cso_dir
#include "../addon/settings.h"     // g_settings.ptskin_test
#include "../core/dxil_cb.h"
#include "../migoto/dx12_pipeline.h"     // MigotoPort::State, OwnedTransition
#include "../migoto/dx12_input_layout.h" // MigotoPort::DescribeInputLayout
#include "pt_skin_owned.h"               // Owned, Xit
#include "rt_hooks.h"
#include "rt_rings.h"
#include "rt_pal_record.h"
#include "rt_diag_dispatch.h"
#include "rt_blas.h"
#include "rt_draw.h"

RtDrawVbs g_rt_draw_vbs;
RtSkinRings g_rt_rings;
RtPalRecorder g_rt_pal_recorder;
RtSkinRead g_rt_skin_read;
RtBonesLog g_rt_bones;
RtBlasHook g_rt_blas;
RtDrawPass g_rt_draw;

namespace {

// Rings are looked up on every dispatch, so only mod characters' rebuilt geometry is kept, and only this many.
constexpr size_t kRingCap = 512;
// RTGEO lines: new rings no mod owns (scenery, other characters), and builds whose index buffer is a mod's.
// Each has its own budget, so early scenery cannot use up the lines a mod character needs.
// Counted per build; a build logs one line per input. Small rebuilt meshes (plants, cloth bits) are not logged.
constexpr uint32_t kGeoLogCapOther = 24;
constexpr uint32_t kGeoLogCapMod = 8;
constexpr uint32_t kCharSizedVertices = 2000;
// RTRINGS: a summary of the recorded rings, at most this often.
constexpr uint32_t kSummaryFrames = 600;

std::atomic<uint32_t> s_geo_logged_other{ 0 };
std::atomic<uint32_t> s_geo_logged_mod{ 0 };
std::atomic<uint32_t> s_last_summary_frame{ 0 };
std::atomic<bool> s_cap_reported{ false };
std::atomic<uint64_t> s_builds_seen{ 0 };
std::atomic<uint64_t> s_dynamic_builds{ 0 };

uint64_t NsHash(int mod)
{
    return mod < 0 ? 0 : (uint64_t)mod + 1; // 0 = no mod known
}

} // namespace

// ---- RtDrawVbs: the vb0 a claimed draw of (mod, comp) last read ----

void RtDrawVbs::Note(int mod_id, int comp_idx, const RtDrawVb& v)
{
    const uint64_t key = ((uint64_t)(uint32_t)mod_id << 32) | (uint32_t)comp_idx;
    std::lock_guard lock(mutex_);
    by_comp_[key] = v;
}

bool RtDrawVbs::Of(int mod_id, int comp_idx, RtDrawVb* out) const
{
    const uint64_t key = ((uint64_t)(uint32_t)mod_id << 32) | (uint32_t)comp_idx;
    std::lock_guard lock(mutex_);
    const auto it = by_comp_.find(key);
    if (it == by_comp_.end()) return false;
    if (out) *out = it->second;
    return true;
}

// ---- RtSkinRings: the vertex buffers skinned BLAS are built from ----

void RtSkinRings::Note(uint64_t va, uint64_t bytes, int mod)
{
    if (!va || !bytes) return;
    std::lock_guard lock(mutex_);
    for (Ring& ring : rings_) {
        if (ring.va == va && ring.bytes == bytes) {
            if (mod >= 0) ring.ns_hash = NsHash(mod);
            return;
        }
    }
    if (rings_.size() >= kRingCap) {
        if (g_debug && !s_cap_reported.exchange(true))
            logger::write("RTRINGS: %zu rings recorded, the cap -- later rebuilt BLAS are not watched", kRingCap);
        return;
    }
    rings_.push_back({ va, bytes, NsHash(mod) });
    ++gen_;
}

bool RtSkinRings::Overlaps(uint64_t va, uint64_t bytes) const
{
    if (!va) return false;
    const uint64_t end = va + (bytes ? bytes : 1);
    std::lock_guard lock(mutex_);
    for (const Ring& ring : rings_)
        if (va < ring.va + ring.bytes && ring.va < end) return true;
    return false;
}

int RtSkinRings::ModOf(uint64_t va, uint64_t bytes, int* overlap_mod) const
{
    if (overlap_mod) *overlap_mod = -1;
    const uint64_t end = va + (bytes ? bytes : 1);
    std::lock_guard lock(mutex_);
    for (const Ring& ring : rings_) {
        const int mod = ring.ns_hash ? (int)(ring.ns_hash - 1) : -1;
        if (ring.va == va && ring.bytes == bytes) return mod;
        if (overlap_mod && *overlap_mod < 0 && va < ring.va + ring.bytes && ring.va < end) *overlap_mod = mod;
    }
    return -1;
}

size_t RtSkinRings::Count() const
{
    std::lock_guard lock(mutex_);
    return rings_.size();
}

void RtSkinRings::ForgetRange(uint64_t va, uint64_t bytes)
{
    std::lock_guard lock(mutex_);
    const uint64_t end = va + bytes;
    std::vector<Ring> kept;
    kept.reserve(rings_.size());
    for (const Ring& ring : rings_) {
        const bool inside = ring.va >= va && ring.va + ring.bytes <= end;
        if (!inside) kept.push_back(ring);
    }
    rings_.swap(kept);
}

uint32_t RtSkinRings::Gen() const
{
    std::lock_guard lock(mutex_);
    return gen_;
}

/**
 * Writes one RTRINGS line: how many rings are recorded and how many belong to each mod.
 * Called from the BLAS hook; it writes at most once every kSummaryFrames frames.
 */
static void LogRingSummary()
{
    const uint32_t frame = g_frame_id;
    uint32_t last = s_last_summary_frame.load(std::memory_order_relaxed);
    if (frame - last < kSummaryFrames) return;
    if (!s_last_summary_frame.compare_exchange_strong(last, frame)) return;
    std::map<int, size_t> per_mod;
    size_t total = 0;
    g_rt_rings.CountPerMod(&per_mod, &total);
    std::string text;
    for (const auto& entry : per_mod) {
        char part[48];
        snprintf(part, sizeof(part), " mod%d=%zu", entry.first, entry.second);
        text += part;
    }
    logger::write("RTRINGS frame=%u: %zu rings, owned by a mod:%s", frame, total, text.empty() ? " none" : text.c_str());
}

void RtSkinRings::CountPerMod(std::map<int, size_t>* per_mod, size_t* total) const
{
    std::lock_guard lock(mutex_);
    *total = rings_.size();
    for (const Ring& ring : rings_)
        if (ring.ns_hash) ++(*per_mod)[(int)(ring.ns_hash - 1)];
}

// ---- RTSCENE (debug): where a mod character sits in the game's ray-tracing scene ----

/** A BLAS the game builds from a mod character's mesh. */
struct ModBlas {
    int mod = -1;
    uint32_t geometries = 0;
};

/** The scene instance that uses a mod character's BLAS. */
struct ModInstance {
    int mod = -1;
    uint32_t geometries = 0;
    uint32_t hit_group_index = 0; // InstanceContributionToHitGroupIndex
};

std::mutex s_scene_mutex;
std::unordered_map<uint64_t, ModBlas> s_mod_blas; // by the BLAS's address
ModInstance s_mod_instance;
std::atomic<int> s_scene_logs{ 0 };
std::atomic<int> s_instance_logs{ 0 };
std::atomic<int> s_records_logs{ 0 };

/**
 * Debug: logs the game's scene build, and each instance in it that uses a mod character's BLAS.
 *
 * @param input_count How many inputs the build has.
 * @param inputs      The inputs; the instance lists are read from the game's upload buffers.
 */
void LogSceneInstances(uint32_t input_count, const reshade::api::acceleration_structure_build_input* inputs)
{
    for (uint32_t i = 0; i < input_count; ++i) {
        const auto& in = inputs[i];
        if (in.type != reshade::api::acceleration_structure_build_input_type::instances) continue;
        ID3D12Resource* resource = reinterpret_cast<ID3D12Resource*>(in.instances.buffer.handle);
        const uint64_t va = RtBlasHook::Va(in.instances.buffer, in.instances.offset);
        const uint32_t count = in.instances.count < 16384 ? in.instances.count : 16384;
        std::vector<uint8_t> bytes((size_t)count * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
        const uint32_t read = resource && !in.instances.array_of_pointers && !bytes.empty()
                                  ? g_game_cbs.Snapshot(resource, va, bytes.data(), (uint32_t)bytes.size())
                                  : 0;
        if (s_scene_logs.fetch_add(1) < 6)
            logger::write("RTSCENE build: %u instances at %llx (resource %s, pointers %d): %u bytes readable on the CPU",
                in.instances.count, (unsigned long long)va, resource ? "known" : "unknown",
                (int)in.instances.array_of_pointers, read);
        std::lock_guard lock(s_scene_mutex);
        for (uint32_t k = 0; (k + 1) * sizeof(D3D12_RAYTRACING_INSTANCE_DESC) <= read; ++k) {
            D3D12_RAYTRACING_INSTANCE_DESC desc;
            memcpy(&desc, bytes.data() + (size_t)k * sizeof(desc), sizeof(desc));
            const auto found = s_mod_blas.find(desc.AccelerationStructure);
            if (found == s_mod_blas.end()) continue;
            s_mod_instance.mod = found->second.mod;
            s_mod_instance.geometries = found->second.geometries;
            s_mod_instance.hit_group_index = desc.InstanceContributionToHitGroupIndex;
            if (s_instance_logs.fetch_add(1) < 12)
                logger::write("RTSCENE instance %u of %u: mod=%d blas=%llx (%u geometries) id=%u mask=%02x "
                              "hit_group_index=%u flags=%x frame=%u",
                    k, in.instances.count, found->second.mod, (unsigned long long)desc.AccelerationStructure,
                    found->second.geometries, (unsigned)desc.InstanceID, (unsigned)desc.InstanceMask,
                    (unsigned)desc.InstanceContributionToHitGroupIndex, (unsigned)desc.Flags, g_frame_id);
        }
    }
}

/**
 * Debug: logs the hit-group records of the last mod instance seen in the scene, as hex words.
 *
 * @param hit_groups The game's hit-group table.
 * @param offset     Where the table starts in it.
 * @param size       The table's size in bytes.
 * @param stride     One record's size in bytes.
 * @param width      The dispatch's width.
 * @param height     The dispatch's height.
 */
void RtNoteDispatchRays(ID3D12Resource* hit_groups, uint64_t offset, uint64_t size, uint64_t stride, uint32_t width,
    uint32_t height)
{
    ModInstance instance;
    {
        std::lock_guard lock(s_scene_mutex);
        instance = s_mod_instance;
    }
    if (instance.mod < 0 || !hit_groups || !stride || stride > 512) return;
    if (s_records_logs.fetch_add(1) >= 4) return;
    const uint64_t table_va = hit_groups->GetGPUVirtualAddress() + offset;
    logger::write("RTSCENE rays %ux%u: hit groups at %llx, %llu bytes, %llu per record; mod=%d starts at record %u",
        width, height, (unsigned long long)table_va, (unsigned long long)size, (unsigned long long)stride, instance.mod,
        instance.hit_group_index);
    // The game may use several records per geometry (one per ray type), so a few more than the geometry count.
    const uint32_t records = instance.geometries * 2 < 24 ? instance.geometries * 2 : 24;
    for (uint32_t r = 0; r < records; ++r) {
        const uint64_t record_offset = (uint64_t)(instance.hit_group_index + r) * stride;
        if (record_offset + stride > size) break;
        uint8_t bytes[512];
        const uint32_t read = g_game_cbs.Snapshot(hit_groups, table_va + record_offset, bytes, (uint32_t)stride);
        std::string words;
        for (uint32_t b = 0; b + 8 <= read; b += 8) {
            uint64_t word = 0;
            memcpy(&word, bytes + b, 8);
            char part[24];
            snprintf(part, sizeof(part), " %llx", (unsigned long long)word);
            words += part;
        }
        logger::write("RTSCENE record +%u (%u bytes read):%s", r, read, words.c_str());
    }
}

// ---- RTHIDE: what a mod hides of the game's own geometry is cut out of the game's ray-tracing geometry ----
//
// The game traces rays against its own mesh. Where a mod skips one of the game's draws without drawing anything
// (a hidden tail), or draws only some index ranges of it (a hidden hair strand), reflections still showed the
// part. Such a BLAS is built here in the game's place, with the hidden triangles' indices set to 0.

/** What a mod's component section leaves of the game's draw it replaces. */
struct ComponentCut {
    uint32_t index_count = 0; // the game's draw
    bool cut = false; // false = the game's geometry stays whole
    std::vector<RtDrawnRange> keep; // index ranges that stay, counted from the draw's first index
    uint64_t signature = 0; // changes when `cut` or `keep` do
};

/** One input of a BLAS built here: the index buffer with the hidden triangles' indices set to 0. */
struct CutInput {
    Microsoft::WRL::ComPtr<ID3D12Resource> indices;
    uint64_t signature = 0;
    uint64_t source_va = 0;
};

/** A BLAS of the game's that is built here every frame. */
struct CutBlas {
    Microsoft::WRL::ComPtr<ID3D12Resource> scratch;
    uint64_t scratch_bytes = 0;
    std::vector<CutInput> inputs;
    uint32_t last_frame = 0;
};

constexpr uint32_t kCutRetireFrames = 32; // a replaced buffer is kept this long: lists in flight may still read it
constexpr uint32_t kCutIdleFrames = 600; // a BLAS not built for this long is forgotten
constexpr uint32_t kCutMaxInputs = 64;

std::mutex s_cut_mutex; // guards everything below
std::map<std::pair<int, uint32_t>, ComponentCut> s_cuts; // by mod and the game draw's first index
/** An index buffer as the game binds it for its draws. */
struct GameIndexBuffer {
    ID3D12Resource* res = nullptr;
    uint32_t bytes = 0;
};
std::map<uint64_t, GameIndexBuffer> s_game_index_buffers; // by the address the game binds
std::unordered_map<uint64_t, CutBlas> s_cut_blas; // by the BLAS's address
std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Resource>, uint32_t>> s_cut_retired;
std::atomic<int> s_cut_logs{ 0 };

/**
 * Mixes a value into a signature.
 *
 * @param signature The signature so far.
 * @param value     The value to add.
 * @return The new signature.
 */
uint64_t MixSignature(uint64_t signature, uint64_t value)
{
    return (signature ^ value) * 1099511628211ull;
}

void RtNoteComponentDraw(int mod, ID3D12Resource* game_ib, uint64_t game_ib_va, uint32_t game_ib_bytes,
    uint32_t game_first_index, uint32_t game_index_count, bool skip_original, const RtDrawnRange* drawn,
    uint32_t drawn_count)
{
    // Cut only when the game's draw is skipped and what is drawn instead lies inside the game's own index range
    // and is less than all of it: then the mod draws the game's geometry in part (or not at all). Anything else
    // is the mod's own geometry, which the game's ray-tracing mesh does not have.
    bool cut = skip_original;
    uint64_t total = 0;
    uint64_t signature = MixSignature(14695981039346656037ull, game_index_count);
    for (uint32_t i = 0; i < drawn_count && cut; ++i) {
        const uint64_t end = (uint64_t)drawn[i].first + drawn[i].count;
        if (drawn[i].first < game_first_index || end > (uint64_t)game_first_index + game_index_count) cut = false;
        total += drawn[i].count;
        signature = MixSignature(MixSignature(signature, drawn[i].first), drawn[i].count);
    }
    if (total >= game_index_count) cut = false;
    if (!cut) signature = 1;

    std::lock_guard lock(s_cut_mutex);
    if (game_ib && game_ib_va) s_game_index_buffers[game_ib_va] = GameIndexBuffer{ game_ib, game_ib_bytes };
    ComponentCut& entry = s_cuts[{ mod, game_first_index }];
    if (entry.signature == signature) return;
    entry.index_count = game_index_count;
    entry.cut = cut;
    entry.signature = signature;
    entry.keep.clear();
    for (uint32_t i = 0; i < drawn_count && cut; ++i)
        entry.keep.push_back(RtDrawnRange{ drawn[i].count, drawn[i].first - game_first_index });
    if (g_debug && s_cut_logs.fetch_add(1) < 40)
        logger::write("RTHIDE mod=%d first_index=%u (%u indices): %s, %zu range(s) kept (frame %u)", mod,
            game_first_index, game_index_count, cut ? "cut from the game's ray-tracing geometry" : "left whole",
            entry.keep.size(), g_frame_id);
}

/**
 * Releases what RTHIDE no longer needs. Call with s_cut_mutex held.
 */
void PurgeCutBuffers()
{
    for (size_t i = 0; i < s_cut_retired.size();) {
        if (g_frame_id > s_cut_retired[i].second + kCutRetireFrames) {
            s_cut_retired[i] = std::move(s_cut_retired.back());
            s_cut_retired.pop_back();
        } else {
            ++i;
        }
    }
    for (auto it = s_cut_blas.begin(); it != s_cut_blas.end();) {
        if (g_frame_id > it->second.last_frame + kCutIdleFrames) {
            if (it->second.scratch) s_cut_retired.push_back({ it->second.scratch, g_frame_id });
            for (CutInput& input : it->second.inputs)
                if (input.indices) s_cut_retired.push_back({ input.indices, g_frame_id });
            it = s_cut_blas.erase(it);
        } else {
            ++it;
        }
    }
}

/**
 * Finds the index buffer, as the game's draws bind it, that holds an address. Call with s_cut_mutex held.
 *
 * @param index_va The address of some indices.
 * @param bound_va Receives the address the game binds the buffer at.
 * @return The buffer; its `res` is null when no draw has bound a buffer holding the address.
 */
GameIndexBuffer GameIndexBufferAt(uint64_t index_va, uint64_t* bound_va)
{
    auto it = s_game_index_buffers.upper_bound(index_va);
    if (it == s_game_index_buffers.begin()) return {};
    --it;
    if (index_va - it->first >= it->second.bytes) return {};
    *bound_va = it->first;
    return it->second;
}

/**
 * Makes the index buffer of one cut input: zeros, with the kept ranges copied in from the game's index buffer.
 *
 * @param cl          The game's command list.
 * @param device      The device.
 * @param game_ib     The game's index buffer.
 * @param game_offset Where the input's indices start in it, in bytes.
 * @param index_size  Bytes per index.
 * @param cut         What stays of the input.
 * @return The buffer, readable by the build; null when it could not be made.
 */
Microsoft::WRL::ComPtr<ID3D12Resource> MakeCutIndices(ID3D12GraphicsCommandList* cl, ID3D12Device* device,
    ID3D12Resource* game_ib, uint64_t game_offset, uint32_t index_size, const ComponentCut& cut)
{
    // The copy needs the game's buffer as a copy source; its state is known only from the game's own barriers.
    const uint32_t state = g_resources.StateOf(game_ib);
    if (state == MigotoPort::kResStateUnknown) return nullptr;
    // A new committed buffer is all zeros: index 0 three times is a triangle no ray can hit.
    Microsoft::WRL::ComPtr<ID3D12Resource> indices = WwmiMakeBuffer(device, (uint64_t)cut.index_count * index_size,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!indices) return nullptr;
    const D3D12_RESOURCE_STATES before = (D3D12_RESOURCE_STATES)state;
    const bool readable = (before & D3D12_RESOURCE_STATE_COPY_SOURCE) != 0;
    if (!cut.keep.empty() && !readable) TransitionBarrier(cl, game_ib, before, before | D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (const RtDrawnRange& range : cut.keep)
        cl->CopyBufferRegion(indices.Get(), (uint64_t)range.first * index_size, game_ib,
            game_offset + (uint64_t)range.first * index_size, (uint64_t)range.count * index_size);
    if (!cut.keep.empty() && !readable) TransitionBarrier(cl, game_ib, before | D3D12_RESOURCE_STATE_COPY_SOURCE, before);
    TransitionBarrier(cl, indices.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return indices;
}

/**
 * Builds a mod character's BLAS in the game's place when the mod hides some of the game's geometry. Once a BLAS
 * has been built here it always is: the game only updates it, and an update keeps the triangles of the last build.
 *
 * @param cmd_list    The game's command list (ReShade's).
 * @param mod         The mod the BLAS belongs to.
 * @param build_flags The game's build flags.
 * @param input_count How many inputs the build has.
 * @param inputs      The game's inputs.
 * @param dest        The BLAS.
 * @return true when the BLAS was built here, so the game's build must not run.
 */
bool BuildCutBlas(reshade::api::command_list* cmd_list, int mod, reshade::api::acceleration_structure_build_flags build_flags,
    uint32_t input_count, const reshade::api::acceleration_structure_build_input* inputs, reshade::api::resource_view dest)
{
    if (g_settings.ptskin_test == 4 || !cmd_list || !dest.handle || !input_count || input_count > kCutMaxInputs)
        return false;
    for (uint32_t i = 0; i < input_count; ++i)
        if (inputs[i].type != reshade::api::acceleration_structure_build_input_type::triangles) return false;

    std::lock_guard lock(s_cut_mutex);
    const ComponentCut* cuts[kCutMaxInputs] = {};
    bool any_cut = false;
    for (uint32_t i = 0; i < input_count; ++i) {
        // The draw the input belongs to: its first index, counted from where the game's draws bind the buffer.
        const uint32_t index_size = inputs[i].triangles.index_format == reshade::api::format::r16_uint ? 2u : 4u;
        const uint64_t index_va = RtBlasHook::Va(inputs[i].triangles.index_buffer, inputs[i].triangles.index_offset);
        uint64_t bound_va = 0;
        const bool bound = GameIndexBufferAt(index_va, &bound_va).res != nullptr;
        const uint32_t first = bound ? (uint32_t)((index_va - bound_va) / index_size) : 0;
        static std::atomic<int> s_match_logs{ 0 };
        if (g_debug && s_match_logs.fetch_add(1) < 24)
            logger::write("RTHIDE match: mod=%d input %u indices at %llx (%u): %s, first_index=%u", mod, i,
                (unsigned long long)index_va, inputs[i].triangles.index_count,
                bound ? "inside an index buffer the game draws with" : "in no index buffer a draw has bound", first);
        if (!bound) continue;
        const auto found = s_cuts.find({ mod, first });
        if (found == s_cuts.end() || !found->second.cut || found->second.index_count != inputs[i].triangles.index_count)
            continue;
        cuts[i] = &found->second;
        any_cut = true;
    }
    auto blas_found = s_cut_blas.find(dest.handle);
    if (!any_cut && blas_found == s_cut_blas.end()) return false;

    ID3D12GraphicsCommandList* cl = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native());
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cl4;
    Microsoft::WRL::ComPtr<ID3D12Device5> device;
    if (!cl || FAILED(cl->QueryInterface(IID_PPV_ARGS(&cl4))) || FAILED(cl->GetDevice(IID_PPV_ARGS(&device)))) return false;

    if (blas_found == s_cut_blas.end()) {
        PurgeCutBuffers();
        blas_found = s_cut_blas.emplace(dest.handle, CutBlas{}).first;
        if (g_debug)
            logger::write("RTHIDE: the BLAS at %llx of mod %d (%u inputs) is built by the add-on from now on (frame %u)",
                (unsigned long long)dest.handle, mod, input_count, g_frame_id);
    }
    CutBlas& blas = blas_found->second;
    blas.last_frame = g_frame_id;
    blas.inputs.resize(input_count);

    D3D12_RAYTRACING_GEOMETRY_DESC geometries[kCutMaxInputs] = {};
    for (uint32_t i = 0; i < input_count; ++i) {
        const auto& triangles = inputs[i].triangles;
        const uint32_t index_size = triangles.index_format == reshade::api::format::r16_uint ? 2u : 4u;
        const uint64_t index_va = RtBlasHook::Va(triangles.index_buffer, triangles.index_offset);
        CutInput& input = blas.inputs[i];
        uint64_t use_index_va = index_va;
        if (cuts[i]) {
            if (!input.indices || input.signature != cuts[i]->signature || input.source_va != index_va) {
                uint64_t bound_va = 0;
                ID3D12Resource* game_ib = GameIndexBufferAt(index_va, &bound_va).res;
                Microsoft::WRL::ComPtr<ID3D12Resource> made;
                if (game_ib && g_resources.Live(game_ib))
                    made = MakeCutIndices(cl, device.Get(), game_ib, index_va - game_ib->GetGPUVirtualAddress(),
                        index_size, *cuts[i]);
                if (input.indices) s_cut_retired.push_back({ input.indices, g_frame_id });
                input.indices = made;
                input.signature = cuts[i]->signature;
                input.source_va = index_va;
                if (g_debug && s_cut_logs.fetch_add(1) < 40)
                    logger::write("RTHIDE: input %u of the BLAS at %llx: %s (%zu range(s) of %u indices kept)", i,
                        (unsigned long long)dest.handle,
                        made ? "cut index buffer made" : "no cut -- the game's index buffer is not known or its state is not",
                        cuts[i]->keep.size(), cuts[i]->index_count);
            }
            if (input.indices) use_index_va = input.indices->GetGPUVirtualAddress();
        } else if (input.indices) {
            s_cut_retired.push_back({ input.indices, g_frame_id });
            input = CutInput{};
        }
        D3D12_RAYTRACING_GEOMETRY_DESC& geometry = geometries[i];
        geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometry.Flags = (D3D12_RAYTRACING_GEOMETRY_FLAGS)inputs[i].flags;
        geometry.Triangles.Transform3x4 = RtBlasHook::Va(triangles.transform_buffer, triangles.transform_offset);
        geometry.Triangles.IndexFormat = (DXGI_FORMAT)triangles.index_format;
        geometry.Triangles.VertexFormat = (DXGI_FORMAT)triangles.vertex_format;
        geometry.Triangles.IndexCount = triangles.index_count;
        geometry.Triangles.VertexCount = triangles.vertex_count;
        geometry.Triangles.IndexBuffer = use_index_va;
        geometry.Triangles.VertexBuffer.StartAddress = RtBlasHook::Va(triangles.vertex_buffer, triangles.vertex_offset);
        geometry.Triangles.VertexBuffer.StrideInBytes = triangles.vertex_stride;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
    desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    desc.Inputs.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
        (uint32_t)build_flags & ~(uint32_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE);
    desc.Inputs.NumDescs = input_count;
    desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    desc.Inputs.pGeometryDescs = geometries;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizes = {};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&desc.Inputs, &sizes);
    if (!sizes.ScratchDataSizeInBytes) return false;
    if (!blas.scratch || blas.scratch_bytes < sizes.ScratchDataSizeInBytes) {
        if (blas.scratch) s_cut_retired.push_back({ blas.scratch, g_frame_id });
        blas.scratch = WwmiMakeBuffer(device.Get(), sizes.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        blas.scratch_bytes = blas.scratch ? sizes.ScratchDataSizeInBytes : 0;
        if (!blas.scratch) return false;
    }
    desc.DestAccelerationStructureData = dest.handle;
    desc.ScratchAccelerationStructureData = blas.scratch->GetGPUVirtualAddress();
    cl4->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
    RemovalPin::Ours(cl, "rt-cut-blas");
    return true;
}

// ---- RtBlasHook: what the game builds its BLAS from ----

uint64_t RtBlasHook::Va(reshade::api::resource r, uint64_t off)
{
    // ReShade passes an address it cannot map to a resource as offset with a null resource.
    if (!r.handle) return off;
    return reinterpret_cast<ID3D12Resource*>(r.handle)->GetGPUVirtualAddress() + off;
}

int RtBlasHook::NameTriangles(const reshade::api::acceleration_structure_build_input& in, uint32_t* first_out)
{
    if (first_out) *first_out = 0;
    if (in.type != reshade::api::acceleration_structure_build_input_type::triangles) return -1;
    const uint64_t index_va = Va(in.triangles.index_buffer, in.triangles.index_offset);
    if (!index_va || !in.triangles.index_count) return -1;
    const uint32_t index_size = in.triangles.index_format == reshade::api::format::r16_uint ? 2u : 4u;
    int mod = -1;
    uint64_t ib_base = 0;
    if (!g_recognition.IbAnchorResolve(index_va, in.triangles.index_count * index_size, &mod, &ib_base)) return -1;
    if (first_out && ib_base <= index_va) *first_out = (uint32_t)((index_va - ib_base) / index_size);
    return mod;
}

bool RtBlasHook::OnBuild(reshade::api::command_list* cmd_list, reshade::api::acceleration_structure_type type,
    reshade::api::acceleration_structure_build_flags build_flags, uint32_t input_count,
    const reshade::api::acceleration_structure_build_input* inputs, reshade::api::resource_view dest,
    reshade::api::acceleration_structure_build_mode mode, unsigned long long n)
{
    s_builds_seen.fetch_add(1, std::memory_order_relaxed);
    if (type == reshade::api::acceleration_structure_type::top_level && inputs && g_debug)
        LogSceneInstances(input_count, inputs);
    if (type != reshade::api::acceleration_structure_type::bottom_level || !inputs) return false;
    // Skinned geometry is built to be updated, and then updated every frame; static scenery is built once.
    const bool dynamic = mode == reshade::api::acceleration_structure_build_mode::update ||
                         (build_flags & reshade::api::acceleration_structure_build_flags::allow_update) != 0;
    if (!dynamic) return false;
    s_dynamic_builds.fetch_add(1, std::memory_order_relaxed);

    // The BLAS belongs to a mod when any of its inputs draws from a mod's index buffer; all inputs then go to it.
    int blas_mod = -1;
    for (uint32_t i = 0; i < input_count && blas_mod < 0; ++i) blas_mod = NameTriangles(inputs[i], nullptr);
    // Only a mod character's vertex buffers are rings: the core learns the skin-cache shader from the first
    // dispatch that writes a ring, so a ring of a plant or a cloth bit would teach it the wrong shader.
    const uint32_t gen_before = g_rt_rings.Gen();
    for (uint32_t i = 0; i < input_count && blas_mod >= 0; ++i) {
        const auto& in = inputs[i];
        if (in.type != reshade::api::acceleration_structure_build_input_type::triangles) continue;
        g_rt_rings.Note(Va(in.triangles.vertex_buffer, in.triangles.vertex_offset),
            (uint64_t)in.triangles.vertex_count * in.triangles.vertex_stride, blas_mod);
    }
    const bool new_ring = blas_mod >= 0 ? g_rt_rings.Gen() != gen_before : true;
    if (blas_mod >= 0 && BuildCutBlas(cmd_list, blas_mod, build_flags, input_count, inputs, dest)) return true;
    // Debug (ptskin_test = 3): a mod character's ray-tracing geometry stops following the animation, to see
    // which parts of a reflection come from it.
    if (blas_mod >= 0 && g_settings.ptskin_test == 3 &&
        mode == reshade::api::acceleration_structure_build_mode::update)
        return true;
    if (!g_debug) return false;
    if (blas_mod >= 0) {
        std::lock_guard lock(s_scene_mutex);
        s_mod_blas[dest.handle] = ModBlas{ blas_mod, input_count };
    }
    LogRingSummary();

    bool log_it = false;
    if (blas_mod >= 0) {
        log_it = s_geo_logged_mod.fetch_add(1, std::memory_order_relaxed) < kGeoLogCapMod;
    } else if (new_ring) {
        bool char_sized = false;
        for (uint32_t i = 0; i < input_count; ++i)
            if (inputs[i].type == reshade::api::acceleration_structure_build_input_type::triangles &&
                inputs[i].triangles.vertex_count >= kCharSizedVertices)
                char_sized = true;
        if (char_sized) log_it = s_geo_logged_other.fetch_add(1, std::memory_order_relaxed) < kGeoLogCapOther;
    }
    for (uint32_t i = 0; log_it && i < input_count; ++i) {
        const auto& in = inputs[i];
        if (in.type != reshade::api::acceleration_structure_build_input_type::triangles) continue;
        const uint64_t vertex_va = Va(in.triangles.vertex_buffer, in.triangles.vertex_offset);
        uint32_t first = 0;
        const int mod = NameTriangles(in, &first);
        {
            logger::write(
                "RTGEO build #%llu input %u/%u mode=%d flags=%x: vertices va=%llx count=%u stride=%llu fmt=%d | "
                "indices va=%llx count=%u fmt=%d | mod=%d first_index=%u dest=%llx frame=%u%s",
                n + 1, i + 1, input_count, (int)mode, (unsigned)build_flags, (unsigned long long)vertex_va,
                in.triangles.vertex_count, (unsigned long long)in.triangles.vertex_stride,
                (int)in.triangles.vertex_format,
                (unsigned long long)Va(in.triangles.index_buffer, in.triangles.index_offset),
                in.triangles.index_count, (int)in.triangles.index_format, mod < 0 ? blas_mod : mod, first,
                (unsigned long long)dest.handle, g_frame_id, new_ring ? " (new ring)" : "");
        }
    }
    (void)cmd_list;
    return false; // the game's build always runs as it is
}

void RtBlasDiag(reshade::api::command_list*, reshade::api::acceleration_structure_type,
    reshade::api::acceleration_structure_build_flags, uint32_t, const reshade::api::acceleration_structure_build_input*,
    reshade::api::resource_view, reshade::api::acceleration_structure_build_mode, unsigned long long)
{
}

// ---- not written yet: palette recording, skin readback, bone log, the draw pass ----

// ---- RtPalRecorder: which dispatch writes a mod's ring, and what else it binds (step 1: logged only) ----

namespace {

// A shader not yet seen writing a mod's ring is examined in one frame out of this many.
constexpr uint32_t kExamEveryFrames = 120;
// RTPAL lines: the first dispatches of each shader that writes a mod's ring, and how many shaders in all.
constexpr uint32_t kPalLogPerCs = 3;
constexpr uint32_t kPalLogTotal = 24;

std::atomic<uint32_t> s_pal_logged{ 0 };

/** One buffer a dispatch binds: a root argument or a descriptor-table slot. */
struct BoundBuffer {
    uint64_t va;
    uint64_t bytes;
    int param;     // root parameter index
    int slot;      // slot in the table, -1 for a root argument
    uint8_t type;  // RootParam::desc_type for a root argument (2 = SRV, 3 = UAV), 0 for a table slot
};

/**
 * Lists the buffers a dispatch binds through its compute root arguments and descriptor tables.
 *
 * @param cmd_list The command list the dispatch is recorded on.
 * @param state    The list's recorded bindings.
 * @param out      Receives the buffers.
 */
void ListBoundBuffers(reshade::api::command_list* cmd_list, const CmdState& state, std::vector<BoundBuffer>* out)
{
    constexpr int kRootCount = 64;
    constexpr int kTableCount = 16;
    constexpr int kSlotCount = 64;
    for (int p = 0; p < kRootCount; ++p) {
        if (!state.root[p].va) continue;
        out->push_back({ state.root[p].va, g_resources.SizeAtVa(state.root[p].va), p, -1, state.root[p].desc_type });
    }
    reshade::api::device* const device = cmd_list->get_device();
    for (int p = 0; p < kTableCount; ++p) {
        const TableStart start = TableStartOf(device, state.root[p].compute_table());
        if (!start) continue;
        for (int i = 0; i < kSlotCount; ++i) {
            ResInfo info;
            if (!g_descriptors.Find({ start.heap, start.base + (uint32_t)i }, info)) continue;
            if (info.cached_va && info.is_buffer) out->push_back({ info.cached_va, info.size, p, i, 0 });
        }
    }
}

/**
 * Describes one shader register of a dispatch: the buffer bound to it, and for a constant buffer its first bytes.
 *
 * @param cmd_list The command list the dispatch is recorded on.
 * @param state    The list's recorded bindings.
 * @param bind     Where the register sits in the root signature.
 * @param name     The register's name for the log, for example "t3".
 * @param is_cbv   true for a b register: its contents are read too.
 * @param out      The text is appended here.
 */
void DescribeRegister(reshade::api::command_list* cmd_list, const CmdState& state, const RegBind& bind,
    const char* name, bool is_cbv, std::string* out)
{
    if (bind.param == 0xFF || bind.param >= 64) return;
    uint64_t va = 0;
    uint64_t bytes = 0;
    ID3D12Resource* res = nullptr;
    if (bind.is_table) {
        const TableStart start = TableStartOf(cmd_list->get_device(), state.root[bind.param].compute_table());
        ResInfo info;
        if (!start || !g_descriptors.Find({ start.heap, start.base + bind.offset }, info)) return;
        va = info.cached_va;
        bytes = info.size;
        res = info.res;
    } else if (is_cbv) {
        const RootCbvBind& cbv = g_command_lists.Cbv(cmd_list).binds[bind.param];
        va = cbv.va;
        bytes = cbv.size;
        res = cbv.res;
    } else {
        va = state.root[bind.param].va;
        bytes = g_resources.SizeAtVa(va);
        res = state.root[bind.param].res;
    }
    if (!va) return;
    char part[160];
    snprintf(part, sizeof(part), " %s=%s%u[%u] %llx/%llu", name, bind.is_table ? "table p" : "root p", bind.param,
        bind.offset, (unsigned long long)va, (unsigned long long)bytes);
    *out += part;
    int ring_overlap = -1;
    if (g_rt_rings.ModOf(va, bytes, &ring_overlap) >= 0 || ring_overlap >= 0) *out += "<RING";
    if (is_cbv && res) {
        uint8_t cb[128];
        const uint32_t got = g_game_cbs.Snapshot(res, va, cb, sizeof(cb), true);
        if (got) {
            *out += "{";
            for (uint32_t d = 0; d + 4 <= got; d += 4) {
                uint32_t value;
                memcpy(&value, cb + d, 4);
                snprintf(part, sizeof(part), "%s%x", d ? "," : "", value);
                *out += part;
            }
            *out += "}";
        }
    }
}

/**
 * Lists a dispatch's bindings by shader register (t#, u#, b#) through its root signature's register map.
 *
 * @param cmd_list The command list the dispatch is recorded on.
 * @param state    The list's recorded bindings.
 * @param out      Receives the text; left empty when the root signature is not known.
 */
/** Writes every compute register of a root signature's register map into the log text. */
struct RegisterDescriber {
    reshade::api::command_list* cmd_list;
    const CmdState* state;
    std::string* text;

    /**
     * @param map The root signature's register map.
     */
    void operator()(const LayoutRegMap& map) const
    {
        char name[8];
        for (uint32_t r = 0; r < sizeof(map.t_cs) / sizeof(map.t_cs[0]) && r < 32; ++r) {
            snprintf(name, sizeof(name), "t%u", r);
            DescribeRegister(cmd_list, *state, map.t_cs[r], name, false, text);
        }
        for (uint32_t r = 0; r < sizeof(map.u_cs) / sizeof(map.u_cs[0]) && r < 16; ++r) {
            snprintf(name, sizeof(name), "u%u", r);
            DescribeRegister(cmd_list, *state, map.u_cs[r], name, false, text);
        }
        for (uint32_t r = 0; r < sizeof(map.b_cs) / sizeof(map.b_cs[0]) && r < 16; ++r) {
            snprintf(name, sizeof(name), "b%u", r);
            DescribeRegister(cmd_list, *state, map.b_cs[r], name, true, text);
        }
    }
};

void DescribeByRegister(reshade::api::command_list* cmd_list, const CmdState& state, std::string* out)
{
    uint64_t layout = g_command_lists.CmpRoot(cmd_list).layout;
    for (int p = 0; p < 64 && !layout; ++p) layout = state.root[p].ctable_note.layout;
    if (!layout) return;
    char head[48];
    snprintf(head, sizeof(head), " layout=%llx:", (unsigned long long)layout);
    std::string text = head;
    if (g_layouts.WithRegMap(layout, RegisterDescriber{ cmd_list, &state, &text })) *out = text;
}

// PTSKIN, further down: records a skin dispatch that writes a mod's ring as one of its sections.
void RecordSkinSection(reshade::api::command_list* cmd_list, const CmdState& state, uint64_t cs_hash);

} // namespace

void RtPalRecorder::Record(reshade::api::command_list* cmd_list, const CmdState& state, uint64_t cs_hash)
{
    if (g_rt_rings.Count() == 0) return;
    RecordSkinSection(cmd_list, state, cs_hash);
    if (!g_debug) return;
    const uint32_t ring_gen = g_rt_rings.Gen();
    {
        std::lock_guard lock(mutex_);
        CsNote& note = by_cs_[cs_hash];
        const bool due = note.writes_ring || note.ring_gen != ring_gen ||
                         g_frame_id - note.exam_frame >= kExamEveryFrames || note.exam_frame == 0;
        if (!due) return;
        note.exam_frame = g_frame_id;
        note.ring_gen = ring_gen;
    }

    std::vector<BoundBuffer> bound;
    ListBoundBuffers(cmd_list, state, &bound);
    int ring_mod = -1;
    size_t ring_index = 0;
    for (size_t k = 0; k < bound.size() && ring_mod < 0; ++k) {
        int overlap = -1;
        const int exact = g_rt_rings.ModOf(bound[k].va, bound[k].bytes, &overlap);
        ring_mod = exact >= 0 ? exact : overlap;
        ring_index = k;
    }
    if (ring_mod < 0) return;

    bool first_time = false;
    {
        std::lock_guard lock(mutex_);
        CsNote& note = by_cs_[cs_hash];
        first_time = !note.writes_ring;
        note.writes_ring = true;
    }
    static std::mutex s_count_mutex;
    static std::unordered_map<uint64_t, uint32_t> s_lines_per_cs;
    {
        std::lock_guard lock(s_count_mutex);
        uint32_t& lines = s_lines_per_cs[cs_hash];
        if (lines >= kPalLogPerCs) return;
        ++lines;
    }
    if (s_pal_logged.fetch_add(1, std::memory_order_relaxed) >= kPalLogTotal) return;

    std::string text;
    DescribeByRegister(cmd_list, state, &text);
    for (size_t k = 0; k < bound.size() && text.empty(); ++k) {
        char part[160];
        const BoundBuffer& b = bound[k];
        if (b.slot < 0)
            snprintf(part, sizeof(part), " | root p%d %s va=%llx bytes=%llu", b.param,
                b.type == 3 ? "UAV" : (b.type == 2 ? "SRV" : "CBV"), (unsigned long long)b.va,
                (unsigned long long)b.bytes);
        else
            snprintf(part, sizeof(part), " | table p%d[%d] va=%llx bytes=%llu", b.param, b.slot,
                (unsigned long long)b.va, (unsigned long long)b.bytes);
        text += part;
        if (k == ring_index) text += " <RING";
    }
    logger::write("RTPAL cs=%016llx frame=%u writes mod %d's ring%s:%s", (unsigned long long)cs_hash, g_frame_id,
        ring_mod, first_time ? " (first seen)" : "", text.c_str());
}
bool RtBonesLog::WindowOpen() const
{
    return false;
}
void RtBonesLog::Log(reshade::api::command_list*, const CmdState&, uint32_t, uint64_t, bool) {}
void RtSkinRead::Drain() {}
void RtSkinRead::Record(reshade::api::command_list*, const CmdState&, uint64_t, uint32_t) {}
// ---- PTSKIN: a mod's own vertices, skinned for the game's ray-tracing draw path ----
//
// With ray tracing on, the game skins a character in Unreal's GPU skin cache (SkinCacheUpdateBatchCS, one dispatch
// per mesh section) and draws it with a pass-through vertex shader that reads the skinned result: position in slot 0,
// tangents (R16G16B16A16_SNORM x2) in slot 1, color in slot 2, UVs in slot 3, the previous position in slot 4. A mod
// binds its bind-pose buffers in the D3D11 layout instead, so its draw comes out in A-pose with broken shading.
//
// Here, each skin dispatch that writes a mod character's BLAS ring is recorded as a section (where it writes, how
// many vertices, its bone palette). At a mod's pass-through draw of component k, RtPtSkin.cso skins all the mod's
// vertices with section k's palette into our own buffers, and the draw's slots are bound to them in the game's layout.

namespace {

/** Where a skin-cache shader keeps one section's values in its cb0, in dwords. */
struct SkinCbLayout {
    uint32_t bytes = 0;     // the cbuffer's declared size
    uint32_t out_start = 0; // SkinCacheStart: the section's first vertex in the output buffer
    uint32_t count = 0;     // NumVertices
    uint32_t scale = 0;     // the float the skinned position is multiplied by
};

/** One section of a mod character as the game skinned it in a frame. */
struct SkinSection {
    uint64_t out_va = 0;      // where the game wrote the section's skinned positions
    uint32_t count = 0;       // its vertices
    uint64_t bones_va = 0;    // its bone palette: 3 float4 rows per bone
    uint64_t bones_bytes = 0; // the palette's size
    ID3D12Resource* bones_res = nullptr; // the buffer the palette lives in
    float scale = 1.0f;
};

/** The sections of one mod character: the frame being recorded, and the last complete one. */
struct ModSections {
    uint32_t frame = 0;
    std::vector<SkinSection> collecting; // this frame's, in dispatch order
    std::vector<SkinSection> last;       // the previous frame's, sorted by out_va
    std::vector<SkinSection> older;      // the one before, sorted by out_va: `last`'s previous frame
    std::string signature; // the last complete list's vertex counts, for the change log
    bool logged = false;
};

/** A buffer range a register resolves to. */
struct BoundRange {
    ID3D12Resource* res = nullptr;
    uint64_t va = 0;
    uint64_t bytes = 0;
};

/** The compute registers PTSKIN reads from a skin dispatch, copied out of a root signature's register map. */
struct SkinRegisterPicker {
    RegBind* bones;
    RegBind* out_positions;
    RegBind* params;
    RegBind* weights;

    /**
     * @param map The root signature's register map.
     */
    void operator()(const LayoutRegMap& map) const
    {
        *bones = map.t_cs[0];
        *out_positions = map.u_cs[1];
        *params = map.b_cs[0];
        *weights = map.t_cs[3];
    }
};

// The vertex formats of the mod buffers PTSKIN reads (the WWMI ini's vb0..vb4) and of the draw it feeds.
constexpr uint32_t kModPositionStride = 12; // vb0 R32G32B32_FLOAT
constexpr uint32_t kModVectorStride = 8;    // vb1 TangentX + TangentZ, R8G8B8A8_SNORM each
constexpr uint32_t kModTexcoordStride = 16; // vb2
constexpr uint32_t kModColorStride = 4;     // vb3 R8G8B8A8_UNORM
constexpr uint32_t kModBlendStride = 16;    // vb4: 8 bone indices, 8 weights
constexpr uint32_t kOutTangentStride = 16;  // our tangents: R16G16B16A16_SNORM x2
// Our outputs at the draw: vertex buffers, and buffer SRVs the vertex shader loads by vertex id.
constexpr D3D12_RESOURCE_STATES kDrawReadState =
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
constexpr uint32_t kBoneBytes = 48;         // 3 float4 rows
constexpr uint32_t kPtSkinMaxBones = 256;         // indices are 8-bit

std::mutex s_skin_cb_mutex;
std::unordered_map<uint64_t, SkinCbLayout> s_skin_cb; // by compute shader hash

// The mesh a skin dispatch works on, told by where its bone-weight input (t3) starts: that buffer is the mesh's
// own and stays, while the output buffer changes (a new one at every character switch). Learned from dispatches
// already known as a mod's, so that later ones are recognised whatever buffer they write.
std::mutex s_mesh_mutex;
std::unordered_map<uint64_t, int> s_mesh_mod; // the weight input's address -> mod
std::atomic<int> s_mesh_logs{ 0 };

std::mutex s_sections_mutex;
std::map<int, ModSections> s_sections; // by mod

std::mutex s_pso_mutex;
std::unordered_map<uint64_t, bool> s_pass_through; // by graphics PSO: does it read the skin cache's layout?

std::mutex s_once_mutex;
std::set<std::pair<int, int>> s_once; // (what, key) pairs already logged

std::atomic<uint64_t> s_ptskin_draws{ 0 };
std::atomic<uint64_t> s_ptskin_skipped{ 0 };

/**
 * Tells whether a message was not logged yet, and marks it logged. Only in debug.
 *
 * @param what Which message.
 * @param key  What it is about, for example a mod and component.
 * @return true the first time for this pair while debug is on.
 */
bool LogOnce(int what, int key)
{
    if (!g_debug) return false;
    std::lock_guard lock(s_once_mutex);
    return s_once.insert({ what, key }).second;
}

/**
 * Gives the cb0 layout of a skin-cache shader variant from the size its DXIL declares.
 *
 * @param bytes The cbuffer's declared size.
 * @param out   Receives the layout.
 * @return false for a size none of the known variants has.
 */
bool SkinCbLayoutForSize(uint32_t bytes, SkinCbLayout* out)
{
    // The variants found in the game (rt_dump): plain, with morph targets, with cloth.
    if (bytes == 28) {
        *out = { 28, 2, 3, 6 };
        return true;
    }
    if (bytes == 32) {
        *out = { 32, 3, 4, 7 };
        return true;
    }
    if (bytes == 168) {
        *out = { 168, 37, 38, 41 };
        return true;
    }
    return false;
}

/**
 * Remembers where a skin-cache shader keeps its section values, read from the cbuffer size its DXIL declares.
 *
 * @param cs_hash   The shader's hash.
 * @param code      Its bytecode.
 * @param code_size Its size in bytes.
 */
void NoteSkinCbLayout(uint64_t cs_hash, const void* code, size_t code_size)
{
    const std::string_view bytes(static_cast<const char*>(code), code_size);
    if (bytes.find("SkinCacheUpdateBatchCS") == std::string_view::npos) return;
    std::vector<DxilCbuf> cbuffers;
    std::string why;
    if (!DxilCbuffers(code, code_size, cbuffers, &why)) {
        logger::write("PTSKIN cs=%016llx: cbuffers not readable (%s)", (unsigned long long)cs_hash, why.c_str());
        return;
    }
    for (const DxilCbuf& cb : cbuffers) {
        if (cb.space != 0 || cb.reg != 0) continue;
        SkinCbLayout layout;
        if (!SkinCbLayoutForSize(cb.bytes, &layout)) {
            logger::write("PTSKIN cs=%016llx: cb0 is %u bytes, not a known skin-cache layout", (unsigned long long)cs_hash,
                cb.bytes);
            return;
        }
        {
            std::lock_guard lock(s_skin_cb_mutex);
            s_skin_cb[cs_hash] = layout;
        }
        logger::write("PTSKIN cs=%016llx: skin-cache shader, cb0 %u bytes (first vertex out at dword %u, count %u, "
                      "scale %u)",
            (unsigned long long)cs_hash, layout.bytes, layout.out_start, layout.count, layout.scale);
        return;
    }
}

/**
 * Gives the size of one element of a typed buffer view.
 *
 * @param format The view's format.
 * @return Its size in bytes, or 0 for a format not listed here.
 */
uint32_t BufferElementBytes(reshade::api::format format)
{
    switch (format) {
    case reshade::api::format::r32g32b32a32_float:
    case reshade::api::format::r32g32b32a32_uint:
    case reshade::api::format::r32g32b32a32_sint:
        return 16;
    case reshade::api::format::r32g32b32_float:
    case reshade::api::format::r32g32b32_uint:
        return 12;
    case reshade::api::format::r32g32_float:
    case reshade::api::format::r32g32_uint:
    case reshade::api::format::r16g16b16a16_float:
    case reshade::api::format::r16g16b16a16_snorm:
    case reshade::api::format::r16g16b16a16_unorm:
    case reshade::api::format::r16g16b16a16_uint:
        return 8;
    case reshade::api::format::r32_float:
    case reshade::api::format::r32_uint:
    case reshade::api::format::r32_sint:
    case reshade::api::format::r32_typeless:
    case reshade::api::format::r8g8b8a8_snorm:
    case reshade::api::format::r8g8b8a8_unorm:
    case reshade::api::format::r8g8b8a8_uint:
    case reshade::api::format::r16g16_float:
    case reshade::api::format::r16g16_snorm:
        return 4;
    default:
        return 0;
    }
}

/**
 * Finds the buffer range a compute register of the current dispatch is bound to.
 *
 * @param cmd_list The command list the dispatch is recorded on.
 * @param state    The list's recorded bindings.
 * @param bind     Where the register sits in the root signature.
 * @param is_cbv   true for a b register.
 * @param out      Receives the range; a table view's own offset and size are applied.
 * @return false when nothing known is bound there.
 */
bool ResolveComputeRegister(reshade::api::command_list* cmd_list, const CmdState& state, const RegBind& bind,
    bool is_cbv, BoundRange* out)
{
    if (bind.param == 0xFF || bind.param >= 64) return false;
    if (bind.is_table) {
        reshade::api::device* const device = cmd_list->get_device();
        const TableStart start = TableStartOf(device, state.root[bind.param].compute_table());
        ResInfo info;
        if (!start || !g_descriptors.Find({ start.heap, start.base + bind.offset }, info) || !info.is_buffer)
            return false;
        out->res = info.res;
        out->va = info.cached_va;
        out->bytes = info.size;
        if (!info.view_handle) return out->va != 0;
        const reshade::api::resource_view_desc desc = device->get_resource_view_desc({ info.view_handle });
        if (desc.type != reshade::api::resource_view_type::buffer) return out->va != 0;
        if (desc.format == reshade::api::format::unknown) { // structured: offset in bytes already
            out->va += desc.buffer.offset;
            out->bytes = (uint64_t)desc.buffer.structured.count * desc.buffer.structured.stride;
            return out->va != 0;
        }
        const uint32_t element = BufferElementBytes(desc.format);
        if (element == 0) return false; // the view's offset cannot be turned into bytes
        out->va += desc.buffer.offset * element; // ReShade gives a typed view's offset and size in elements
        if (desc.buffer.size != UINT64_MAX) out->bytes = desc.buffer.size * element;
        return out->va != 0;
    }
    if (is_cbv) {
        const RootCbvBind& cbv = g_command_lists.Cbv(cmd_list).binds[bind.param];
        out->res = cbv.res;
        out->va = cbv.va;
        out->bytes = cbv.size;
        return out->va != 0;
    }
    out->res = state.root[bind.param].res;
    out->va = state.root[bind.param].va;
    out->bytes = g_resources.SizeAtVa(out->va);
    return out->va != 0;
}

/**
 * Orders sections by where they write, which is the mesh's section order.
 *
 * @param a A section.
 * @param b Another one.
 * @return true when a comes first.
 */
bool SectionBefore(const SkinSection& a, const SkinSection& b)
{
    return a.out_va < b.out_va;
}

/**
 * Records a skin-cache dispatch that writes a mod character's ring as one of its sections.
 *
 * @param cmd_list The command list the dispatch is recorded on.
 * @param state    The list's recorded bindings.
 * @param cs_hash  The dispatch's compute shader.
 */
void CopySectionPalette(reshade::api::command_list* cmd_list, const CmdState& state, int mod,
    const SkinSection& section);

void RecordSkinSection(reshade::api::command_list* cmd_list, const CmdState& state, uint64_t cs_hash)
{
    SkinCbLayout cb_layout;
    {
        std::lock_guard lock(s_skin_cb_mutex);
        const auto found = s_skin_cb.find(cs_hash);
        if (found == s_skin_cb.end()) return;
        cb_layout = found->second;
    }
    uint64_t layout = g_command_lists.CmpRoot(cmd_list).layout;
    if (!layout) layout = g_command_lists.CmpLayout(cmd_list);
    RegBind bones;
    RegBind out_positions;
    RegBind params;
    RegBind weights;
    if (!layout || !g_layouts.WithRegMap(layout, SkinRegisterPicker{ &bones, &out_positions, &params, &weights })) {
        if (LogOnce(1, 0))
            logger::write("PTSKIN cs=%016llx: root signature %llx not known", (unsigned long long)cs_hash,
                (unsigned long long)layout);
        return;
    }
    BoundRange out_range;
    BoundRange bones_range;
    BoundRange params_range;
    const bool have_out = ResolveComputeRegister(cmd_list, state, out_positions, false, &out_range);
    const bool have_bones = ResolveComputeRegister(cmd_list, state, bones, false, &bones_range);
    const bool have_params = ResolveComputeRegister(cmd_list, state, params, true, &params_range);
    if (!have_out || !have_bones || !have_params || !params_range.res) {
        if (LogOnce(2, 0))
            logger::write("PTSKIN cs=%016llx: registers not resolved: u1 %s (p%u%s+%u), t0 %s (p%u%s+%u), b0 %s (p%u%s)",
                (unsigned long long)cs_hash, have_out ? "ok" : "MISSING", out_positions.param,
                out_positions.is_table ? " table" : "", out_positions.offset, have_bones ? "ok" : "MISSING",
                bones.param, bones.is_table ? " table" : "", bones.offset, have_params ? "ok" : "MISSING", params.param,
                params.is_table ? " table" : "");
        return;
    }
    uint32_t cb[48] = {};
    const uint32_t got = g_game_cbs.Snapshot(params_range.res, params_range.va, reinterpret_cast<uint8_t*>(cb),
        cb_layout.bytes, true);
    if (got < cb_layout.bytes) {
        if (LogOnce(3, 0))
            logger::write("PTSKIN cs=%016llx: cb0 at %llx not readable (%u of %u bytes)", (unsigned long long)cs_hash,
                (unsigned long long)params_range.va, got, cb_layout.bytes);
        return;
    }
    SkinSection section;
    section.count = cb[cb_layout.count];
    section.out_va = out_range.va + (uint64_t)cb[cb_layout.out_start] * kModPositionStride;
    memcpy(&section.scale, &cb[cb_layout.scale], sizeof(float));
    section.bones_va = bones_range.va;
    section.bones_bytes = bones_range.bytes;
    section.bones_res = bones_range.res;
    if (section.count == 0) return;
    int overlap = -1;
    const int exact = g_rt_rings.ModOf(section.out_va, (uint64_t)section.count * kModPositionStride, &overlap);
    int mod = exact >= 0 ? exact : overlap;
    BoundRange weights_range;
    if (ResolveComputeRegister(cmd_list, state, weights, false, &weights_range) && weights_range.va) {
        std::lock_guard lock(s_mesh_mutex);
        if (mod >= 0) {
            s_mesh_mod[weights_range.va] = mod;
        } else {
            const auto mesh_found = s_mesh_mod.find(weights_range.va);
            if (mesh_found != s_mesh_mod.end()) {
                mod = mesh_found->second;
                if (g_debug && s_mesh_logs.fetch_add(1) < 24)
                    logger::write("PTSKIN-MESH mod=%d frame=%u: a dispatch into %llx, a buffer not known as the mod's "
                                  "yet, recognised by its weight input at %llx",
                        mod, g_frame_id, (unsigned long long)section.out_va, (unsigned long long)weights_range.va);
            }
        }
    }
    if (mod < 0) return;
    CopySectionPalette(cmd_list, state, mod, section);

    std::lock_guard lock(s_sections_mutex);
    ModSections& sections = s_sections[mod];
    if (sections.frame != g_frame_id) {
        if (!sections.collecting.empty()) {
            sections.older.swap(sections.last);
            sections.last.swap(sections.collecting);
            std::sort(sections.last.begin(), sections.last.end(), SectionBefore);
            std::string signature;
            for (const SkinSection& s : sections.last) {
                signature += std::to_string(s.count) + "/" + std::to_string(s.bones_bytes / kBoneBytes) + " ";
            }
            static std::atomic<int> s_signature_logs{ 0 };
            if (signature != sections.signature) {
                if (g_debug && s_signature_logs.fetch_add(1) < 60)
                    logger::write("PTSKIN-LOD mod=%d frame=%u: %zu sections (vertices/bones): %s", mod,
                        sections.frame, sections.last.size(), signature.c_str());
                sections.signature = signature;
            }
            if (!sections.logged && g_debug) {
                sections.logged = true;
                std::string text;
                for (const SkinSection& s : sections.last) {
                    char part[112];
                    snprintf(part, sizeof(part), " [out %llx, %u vertices, %llu bones, scale %.3f]",
                        (unsigned long long)s.out_va, s.count, (unsigned long long)(s.bones_bytes / kBoneBytes),
                        s.scale);
                    text += part;
                }
                logger::write("PTSKIN-SECTIONS mod=%d frame=%u: %zu sections:%s", mod, sections.frame,
                    sections.last.size(), text.c_str());
            }
        }
        sections.collecting.clear();
        sections.frame = g_frame_id;
    }
    for (SkinSection& known : sections.collecting) {
        if (known.out_va == section.out_va) {
            known = section;
            return;
        }
    }
    sections.collecting.push_back(section);
}

/**
 * Finds a component's section among those the game skinned into the window its own draw binds at vb0. That window
 * is this mesh's output for this frame, so other meshes' sections and the other of the game's two rings fall
 * outside it. A mesh's sections follow one another without a gap, so the comp-th is found by walking from the
 * window's start, each section beginning where the one before ends. Counting the sections seen instead goes wrong
 * while only some of a frame's dispatches are recorded, or when one from another mesh is left over in the window
 * (a character switch): the component then gets another section's bones, and its vertices fly off.
 *
 * @param mod        The mod.
 * @param comp       The component's index.
 * @param window_va  The start of the game draw's vb0.
 * @param window_end Its end.
 * @param out        Receives the section.
 * @param total      Receives how many sections lie in the window.
 * @return false when the window holds no comp-th section.
 */
bool SectionInWindow(int mod, int comp, uint64_t window_va, uint64_t window_end, SkinSection* out, size_t* total)
{
    std::lock_guard lock(s_sections_mutex);
    *total = 0;
    const auto found = s_sections.find(mod);
    if (found == s_sections.end()) return false;
    const ModSections& sections = found->second;
    std::vector<SkinSection> inside;
    const std::vector<SkinSection>* lists[3] = { &sections.collecting, &sections.last, &sections.older };
    for (const std::vector<SkinSection>* list : lists) {
        for (const SkinSection& section : *list) {
            if (section.out_va < window_va || section.out_va >= window_end) continue;
            bool known = false;
            for (const SkinSection& other : inside) {
                if (other.out_va == section.out_va) known = true;
            }
            if (!known) inside.push_back(section); // the newest list comes first, so it wins
        }
    }
    *total = inside.size();
    if (comp < 0) return false;
    uint64_t cursor = window_va;
    for (int index = 0; index <= comp; ++index) {
        const SkinSection* next = nullptr;
        for (const SkinSection& section : inside) {
            if (section.out_va == cursor) next = &section;
        }
        if (!next) return false;
        if (index == comp) *out = *next;
        cursor += (uint64_t)next->count * kModPositionStride;
    }
    return true;
}

/**
 * Tells whether a graphics PSO reads the skin cache's vertex layout (the ray-tracing pass-through draw).
 *
 * @param pso The PSO (a ReShade pipeline handle).
 * @return true for position at s0, tangents at s1 (+8 = TangentZ, R16G16B16A16_SNORM), previous position at s4.
 */
bool IsPassThroughPso(uint64_t pso)
{
    if (!pso) return false;
    {
        std::lock_guard lock(s_pso_mutex);
        const auto found = s_pass_through.find(pso);
        if (found != s_pass_through.end()) return found->second;
    }
    char text[1024] = {};
    MigotoPort::DescribeInputLayout(pso, text, sizeof(text));
    const bool pass_through = strstr(text, "ATTRIBUTE0@s0+0:f6") && strstr(text, "ATTRIBUTE2@s1+8:f13") &&
                              strstr(text, "ATTRIBUTE14@s4+0:f6");
    std::lock_guard lock(s_pso_mutex);
    s_pass_through[pso] = pass_through;
    return pass_through;
}

/**
 * Our compute pipeline (RtPtSkin.cso) and the output buffers of each mod component. Each component has its own:
 * the game records its lists in parallel and the GPU may run lists of one submission side by side, so one buffer
 * shared by a mod's components could be rewritten with another component's bones while a draw still reads it.
 */
struct PtSkinGpu {
    struct Outputs {
        Owned positions;
        Owned tangents;
        Owned previous_positions; // skinned with the previous frame's palette: the draw's slot 4
        Owned colors;             // the mod's colors with red and blue swapped: the game reads B8G8R8A8
        uint32_t vertices = 0;
    };
    // A section's bone palette, copied at the game's skin dispatch (RtPalCopy.cso). The game reuses its own
    // palette memory after the frame, so the mod is skinned from these.
    // One slot per frame, in turn: the depth, base and velocity passes of a frame all read that frame's slot,
    // even when the game records or runs the next frame's skin pass in between. The slot of the frame before
    // gives the motion vectors' previous positions.
    static constexpr int kSlots = 3;
    struct Palette {
        Owned slot[kSlots];
        uint32_t frame[kSlots] = {}; // the frame each slot was copied in
        uint32_t bytes[kSlots] = {}; // how much of it holds a palette; 0 = never copied
    };
    std::mutex mutex;
    bool tried = false;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_signature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> copy_pso; // RtPalCopy.cso, on the same root signature
    std::map<int, Outputs> by_component; // by mod * 256 + component
    // By mod and PaletteKey: the game writes a section to one of two rings in turn, so out_va changes every frame.
    std::map<std::pair<int, uint64_t>, Palette> palettes;
    // Debug only: a readback of one palette and a few skinned vertices, logged some frames later.
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    const uint8_t* readback_data = nullptr;
    uint32_t readback_frame = 0;      // when the copy was recorded; 0 = none waiting
    uint32_t readback_next_frame = 0; // the earliest frame for the next one
    int readback_done = 0;
    int readback_key = 0;
    uint32_t readback_slot_frame = 0;
    uint32_t readback_vertices = 0;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> retired; // outgrown buffers a list in flight may still read
};
PtSkinGpu s_gpu;

/**
 * Creates the PTSKIN root signature and pipeline once. Call with s_gpu.mutex held.
 *
 * @param device The game's device.
 * @return true when they exist.
 */
bool EnsurePtSkinPipeline(ID3D12Device* device)
{
    if (s_gpu.pso) return true;
    if (s_gpu.tried) return false;
    s_gpu.tried = true;
    D3D12_ROOT_PARAMETER params[9] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0; // b0: vertex count, bone count, scale
    params[0].Constants.Num32BitValues = 4;
    for (UINT t = 0; t < 5; ++t) { // t0 bones, t1 positions, t2 vectors, t3 weights, t4 colors
        params[1 + t].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1 + t].Descriptor.ShaderRegister = t;
    }
    for (UINT u = 0; u < 3; ++u) { // u0 positions out, u1 tangents out, u2 colors out
        params[6 + u].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[6 + u].Descriptor.ShaderRegister = u;
    }
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 9;
    desc.pParameters = params;
    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error)) ||
        FAILED(device->CreateRootSignature(
            0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&s_gpu.root_signature)))) {
        logger::write("PTSKIN: root signature not created");
        return false;
    }
    const std::vector<uint8_t> code = WwmiReadFile((g_addon.cso_dir + L"\\RtPtSkin.cso").c_str());
    if (code.empty()) {
        logger::write("PTSKIN: RtPtSkin.cso missing in %S", g_addon.cso_dir.c_str());
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.pRootSignature = s_gpu.root_signature.Get();
    pso_desc.CS.pShaderBytecode = code.data();
    pso_desc.CS.BytecodeLength = code.size();
    if (FAILED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&s_gpu.pso)))) {
        logger::write("PTSKIN: pipeline not created");
        return false;
    }
    const std::vector<uint8_t> copy_code = WwmiReadFile((g_addon.cso_dir + L"\\RtPalCopy.cso").c_str());
    if (copy_code.empty()) {
        logger::write("PTSKIN: RtPalCopy.cso missing in %S", g_addon.cso_dir.c_str());
        s_gpu.pso.Reset();
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC copy_desc = {};
    copy_desc.pRootSignature = s_gpu.root_signature.Get();
    copy_desc.CS.pShaderBytecode = copy_code.data();
    copy_desc.CS.BytecodeLength = copy_code.size();
    if (FAILED(device->CreateComputePipelineState(&copy_desc, IID_PPV_ARGS(&s_gpu.copy_pso)))) {
        logger::write("PTSKIN: palette copy pipeline not created");
        s_gpu.pso.Reset();
        return false;
    }
    logger::write("PTSKIN: pipeline ready");
    return true;
}

/**
 * Gives a mod component's output buffers, large enough for the mod's vertices. Call with s_gpu.mutex held.
 *
 * @param device   The game's device.
 * @param key      The mod and component, as mod * 256 + component.
 * @param vertices The mod's vertex count.
 * @return The buffers, or nullptr when they could not be made.
 */
PtSkinGpu::Outputs* OutputsOf(ID3D12Device* device, int key, uint32_t vertices)
{
    PtSkinGpu::Outputs& outputs = s_gpu.by_component[key];
    if (outputs.vertices >= vertices && outputs.positions.res && outputs.tangents.res &&
        outputs.previous_positions.res && outputs.colors.res)
        return &outputs;
    if (outputs.positions.res) s_gpu.retired.push_back(outputs.positions.res);
    if (outputs.tangents.res) s_gpu.retired.push_back(outputs.tangents.res);
    if (outputs.previous_positions.res) s_gpu.retired.push_back(outputs.previous_positions.res);
    if (outputs.colors.res) s_gpu.retired.push_back(outputs.colors.res);
    outputs = PtSkinGpu::Outputs{};
    outputs.positions.res = WwmiMakeBuffer(device, (UINT64)vertices * kModPositionStride, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    outputs.tangents.res = WwmiMakeBuffer(device, (UINT64)vertices * kOutTangentStride, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    outputs.previous_positions.res = WwmiMakeBuffer(device, (UINT64)vertices * kModPositionStride,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    outputs.colors.res = WwmiMakeBuffer(device, (UINT64)vertices * kModColorStride, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    if (!outputs.positions.res || !outputs.tangents.res || !outputs.previous_positions.res || !outputs.colors.res) {
        s_gpu.by_component.erase(key);
        return nullptr;
    }
    outputs.vertices = vertices;
    return &outputs;
}

/**
 * Puts the game's compute root signature and its arguments back after our dispatch on its list.
 *
 * @param cmd_list The game's command list.
 * @param cl       The same list, native.
 * @return false when the game had not bound a compute root signature on it yet (nothing to put back).
 */
bool RestoreGameComputeRoot(reshade::api::command_list* cmd_list, ID3D12GraphicsCommandList* cl)
{
    const CmdGfxRoot& root = g_command_lists.CmpRoot(cmd_list);
    if (!root.layout) return false;
    cl->SetComputeRootSignature(reinterpret_cast<ID3D12RootSignature*>(root.layout));
    for (UINT p = 0; p < 64; ++p) {
        switch (root.kind[p]) {
        case CmdGfxRoot::kTable:
            cl->SetComputeRootDescriptorTable(p, D3D12_GPU_DESCRIPTOR_HANDLE{ root.value[p] });
            break;
        case CmdGfxRoot::kCbv:
            if (root.value[p]) cl->SetComputeRootConstantBufferView(p, root.value[p]);
            break;
        case CmdGfxRoot::kSrv:
            if (root.value[p]) cl->SetComputeRootShaderResourceView(p, root.value[p]);
            break;
        case CmdGfxRoot::kUav:
            if (root.value[p]) cl->SetComputeRootUnorderedAccessView(p, root.value[p]);
            break;
        case CmdGfxRoot::kNone:
            break;
        }
    }
    for (uint32_t i = 0; i < root.constant_count; ++i)
        cl->SetComputeRoot32BitConstant(root.constants[i].param, root.constants[i].value, root.constants[i].offset);
    return true;
}

/**
 * Names a section the same way every frame: its vertex count and palette size. Its output address is no good,
 * since the game writes each frame to the other of two rings.
 *
 * @param section The section.
 * @return The key.
 */
uint64_t PaletteKey(const SkinSection& section)
{
    return ((uint64_t)section.count << 32) | (section.bones_bytes & 0xFFFFFFFFull);
}

/**
 * Copies a section's bone palette into the add-on's buffers, on the game's compute list right before the game's
 * own skin dispatch for it. The copy made last time becomes the previous palette first. Puts the game's compute
 * pipeline and root arguments back afterwards; does nothing when they are not known.
 *
 * @param cmd_list The game's compute list.
 * @param state    Its bindings.
 * @param mod      The mod whose ring the section writes.
 * @param section  The section, with the game's palette address.
 */
void CopySectionPalette(reshade::api::command_list* cmd_list, const CmdState& state, int mod,
    const SkinSection& section)
{
    uint64_t bytes = section.bones_bytes;
    if (bytes > (uint64_t)kPtSkinMaxBones * kBoneBytes) bytes = (uint64_t)kPtSkinMaxBones * kBoneBytes;
    bytes -= bytes % kBoneBytes;
    if (bytes == 0 || !section.bones_va) return;
    if (!state.cmp_pso || !g_command_lists.CmpRoot(cmd_list).layout) {
        if (LogOnce(20, mod))
            logger::write("PTSKIN mod=%d: the game's compute pipeline is not known on this list, palette not copied",
                mod);
        return;
    }
    ID3D12GraphicsCommandList* cl = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native());
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    if (!cl || FAILED(cl->GetDevice(IID_PPV_ARGS(&device)))) return;

    std::lock_guard lock(s_gpu.mutex);
    if (!EnsurePtSkinPipeline(device.Get())) return;
    PtSkinGpu::Palette& palette = s_gpu.palettes[{ mod, PaletteKey(section) }];
    if (!palette.slot[0].res) {
        const UINT64 size = (UINT64)kPtSkinMaxBones * kBoneBytes;
        for (int i = 0; i < PtSkinGpu::kSlots; ++i) {
            palette.slot[i].res = WwmiMakeBuffer(device.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            if (!palette.slot[i].res) {
                s_gpu.palettes.erase({ mod, PaletteKey(section) });
                return;
            }
        }
    }
    const int index = (int)(g_frame_id % PtSkinGpu::kSlots);
    Owned& slot = palette.slot[index];
    const uint64_t gen = g_command_lists.Gen(cmd_list);
    cl->SetComputeRootSignature(s_gpu.root_signature.Get());
    cl->SetPipelineState(s_gpu.copy_pso.Get());
    Xit(cl, gen, slot, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t constants[4] = { (uint32_t)(bytes / 4), 0, 0, 0 };
    cl->SetComputeRoot32BitConstants(0, 4, constants, 0);
    cl->SetComputeRootShaderResourceView(1, section.bones_va);
    cl->SetComputeRootUnorderedAccessView(6, slot.va());
    cl->Dispatch((uint32_t)((bytes / 4 + 63) / 64), 1, 1);
    // Back to COMMON, so the draw lists can read it whatever they assume at their start.
    Xit(cl, gen, slot, D3D12_RESOURCE_STATE_COMMON);
    palette.frame[index] = g_frame_id;
    palette.bytes[index] = (uint32_t)bytes;
    static std::atomic<int> s_trace_copies{ 0 };
    if (g_debug && s_trace_copies.fetch_add(1) < 64)
        logger::write("PTSKIN-TRACE copy mod=%d frame=%u out=%llx bones=%llx bytes=%llu slot=%d list_type=%d", mod,
            g_frame_id, (unsigned long long)section.out_va, (unsigned long long)section.bones_va,
            (unsigned long long)bytes, index, (int)cl->GetType());

    cl->SetPipelineState(reinterpret_cast<ID3D12PipelineState*>(state.cmp_pso));
    RestoreGameComputeRoot(cmd_list, cl);
}

/**
 * Makes a mod buffer readable by our compute pass, keeping it readable as a vertex buffer.
 *
 * @param cl  The game's command list.
 * @param res The mod buffer.
 */
void ReadyModBufferForSkin(ID3D12GraphicsCommandList* cl, ID3D12Resource* res)
{
    // The port tracks its own buffers' state per list; anything else is promoted from COMMON on first read.
    if (res && MigotoPort::OwnedBuffer(res))
        MigotoPort::OwnedTransition(
            cl, res, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

/**
 * Debug only: logs the readback recorded by SkinDraw once the GPU is surely done with it (10 frames later):
 * the first two bones of the palette and the first four skinned positions. Call with s_gpu.mutex held.
 */
void LogPtSkinReadback()
{
    if (!s_gpu.readback_frame || g_frame_id < s_gpu.readback_frame + 10) return;
    float values[24 + 48];
    memcpy(values, s_gpu.readback_data, sizeof(values));
    std::string samples;
    for (uint32_t i = 0; i < 16; ++i) {
        char part[96];
        snprintf(part, sizeof(part), " v%u (%.2f %.2f %.2f)", (uint32_t)((uint64_t)i * s_gpu.readback_vertices / 16),
            values[24 + i * 3], values[25 + i * 3], values[26 + i * 3]);
        samples += part;
    }
    logger::write("PTSKIN-READBACK mod=%d comp=%d frame=%u (palette of frame %u): bone0 [%.3f %.3f %.3f %.3f | %.3f %.3f "
                  "%.3f %.3f | %.3f %.3f %.3f %.3f] bone1 [%.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f "
                  "%.3f] skinned:%s",
        s_gpu.readback_key / 256, s_gpu.readback_key % 256, s_gpu.readback_frame, s_gpu.readback_slot_frame,
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9],
        values[10], values[11], values[12], values[13], values[14], values[15], values[16], values[17], values[18],
        values[19], values[20], values[21], values[22], values[23], samples.c_str());
    s_gpu.readback_frame = 0;
    s_gpu.readback_done += 1;
    s_gpu.readback_next_frame = g_frame_id + 30;
}

} // namespace

/**
 * The result of a pass-through draw whose mod could not be skinned (no section or palette yet, as in the first
 * frames after a character switch): the draw must not run, since the mod's bind-pose buffers do not fit the layout.
 *
 * @return The result that tells the caller to draw nothing.
 */
VfRestore CannotSkin()
{
    VfRestore result;
    result.skip = true;
    return result;
}

VfRestore RtDrawPass::SkinDraw(reshade::api::command_list* cmd_list, ID3D12GraphicsCommandList* d3d12_cl, int dm,
    const WwmiMod::Component* wc, uint32_t, uint32_t, uint32_t, uint32_t, int32_t vertex_offset, uint64_t, uint64_t, int,
    bool)
{
    if (!wc || !d3d12_cl || dm < 0) return {};
    const CmdState& state = g_command_lists.Gfx(cmd_list);
    if (!IsPassThroughPso(state.gfx_pso)) return {};
    const int comp = wc->index;
    const int once_key = dm * 256 + (comp & 0xFF); // the mod and component, for the log and the output buffers

    const MigotoPort::PipeState& port = MigotoPort::State();
    const MigotoPort::ShimVbRecord& positions = port.shim_vb[0];
    const MigotoPort::ShimVbRecord& vectors = port.shim_vb[1];
    const MigotoPort::ShimVbRecord& texcoords = port.shim_vb[2];
    const MigotoPort::ShimVbRecord& colors = port.shim_vb[3];
    const MigotoPort::ShimVbRecord& blend = port.shim_vb[4];
    if (positions.stride != kModPositionStride || vectors.stride != kModVectorStride ||
        texcoords.stride != kModTexcoordStride || colors.stride != kModColorStride || blend.stride != kModBlendStride ||
        !positions.va || !vectors.va || !texcoords.va || !colors.va || !blend.va || port.layout_override.count) {
        s_ptskin_skipped.fetch_add(1, std::memory_order_relaxed);
        if (LogOnce(10, once_key))
            logger::write("PTSKIN mod=%d comp=%d: not the WWMI vertex layout (vb0..vb4 strides %u/%u/%u/%u/%u, layout "
                          "override %u) -- drawn as it is",
                dm, comp, positions.stride, vectors.stride, texcoords.stride, colors.stride, blend.stride,
                (unsigned)port.layout_override.count);
        return {};
    }
    // Mods' buffers can differ by a few vertices (padding); skin as many as every one of them holds.
    uint32_t vertices = positions.size / kModPositionStride;
    vertices = (std::min)(vertices, (uint32_t)(vectors.size / kModVectorStride));
    vertices = (std::min)(vertices, (uint32_t)(blend.size / kModBlendStride));
    vertices = (std::min)(vertices, (uint32_t)(colors.size / kModColorStride));
    if (vertices == 0) {
        s_ptskin_skipped.fetch_add(1, std::memory_order_relaxed);
        if (LogOnce(11, once_key))
            logger::write("PTSKIN mod=%d comp=%d: buffer sizes do not agree (%u vertices, vb1 %u, vb4 %u bytes)", dm,
                comp, vertices, vectors.size, blend.size);
        return {};
    }
    // The game's own draw binds this mesh's whole skinned output at vb0; its sections lie inside, in order.
    const uint64_t window_va = state.vb[0].va + (int64_t)vertex_offset * kModPositionStride;
    const uint64_t window_end = state.vb[0].va + state.vb[0].size;
    SkinSection section;
    size_t section_total = 0;
    if (!state.vb[0].va || !SectionInWindow(dm, comp, window_va, window_end, &section, &section_total)) {
        // The game's dispatches into this buffer are recorded only once it is known as the mod's. A BLAS build
        // tells that some frames after a character appears; this draw tells it now.
        if (state.vb[0].va) g_rt_rings.Note(state.vb[0].va, state.vb[0].size, dm);
        static std::atomic<int> s_skip_logs{ 0 };
        if (g_debug && comp == 0 && s_skip_logs.fetch_add(1) < 80)
            logger::write("PTSKIN-SKIP mod=%d frame=%u: not drawn, no section yet in vb0 %llx (%zu there)", dm, g_frame_id,
                (unsigned long long)state.vb[0].va, section_total);
        static std::atomic<int> s_broken_logs{ 0 };
        if (g_debug && comp == 0 && section_total > 0 && s_broken_logs.fetch_add(1) < 40)
            logger::write("PTSKIN-CHAIN mod=%d comp=%d frame=%u: %zu section(s) in vb0 %llx but none leads to this "
                          "component from the start -- not drawn rather than skinned with another section's bones",
                dm, comp, g_frame_id, section_total, (unsigned long long)state.vb[0].va);
        s_ptskin_skipped.fetch_add(1, std::memory_order_relaxed);
        if (LogOnce(12, once_key))
            logger::write("PTSKIN mod=%d comp=%d: no section %d among the %zu recorded in the game's vb0 %llx..%llx "
                          "(base vertex %d)",
                dm, comp, comp, section_total, (unsigned long long)window_va, (unsigned long long)window_end,
                vertex_offset);
        return CannotSkin();
    }
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    if (FAILED(d3d12_cl->GetDevice(IID_PPV_ARGS(&device)))) return {};
    std::lock_guard lock(s_gpu.mutex);
    if (!EnsurePtSkinPipeline(device.Get())) return {};
    // The section's palette as copied at the game's skin dispatch: on the GPU, that of the frame being drawn.
    const auto palette_found = s_gpu.palettes.find({ dm, PaletteKey(section) });
    if (palette_found == s_gpu.palettes.end()) {
        s_ptskin_skipped.fetch_add(1, std::memory_order_relaxed);
        if (LogOnce(13, once_key))
            logger::write("PTSKIN mod=%d comp=%d: no copy of the section's bone palette yet", dm, comp);
        return CannotSkin();
    }
    PtSkinGpu::Palette& palette = palette_found->second;
    // This frame's slot, else the newest one; the previous palette is the newest slot older than that.
    int current_index = -1;
    int previous_index = -1;
    for (int i = 0; i < PtSkinGpu::kSlots; ++i) {
        if (palette.bytes[i] == 0 || palette.frame[i] > g_frame_id) continue;
        if (current_index < 0 || palette.frame[i] > palette.frame[current_index]) current_index = i;
    }
    if (current_index < 0) {
        s_ptskin_skipped.fetch_add(1, std::memory_order_relaxed);
        if (LogOnce(13, once_key))
            logger::write("PTSKIN mod=%d comp=%d: no copy of the section's bone palette yet", dm, comp);
        return CannotSkin();
    }
    for (int i = 0; i < PtSkinGpu::kSlots; ++i) {
        if (palette.bytes[i] == 0 || palette.frame[i] >= palette.frame[current_index]) continue;
        if (previous_index < 0 || palette.frame[i] > palette.frame[previous_index]) previous_index = i;
    }
    Owned& current_slot = palette.slot[current_index];
    const uint32_t bone_count = palette.bytes[current_index] / kBoneBytes;
    // The previous palette gives the positions motion vectors are made from; without it, no motion.
    const bool has_previous = previous_index >= 0 && g_settings.ptskin_test != 1;
    const uint32_t previous_bone_count = has_previous ? palette.bytes[previous_index] / kBoneBytes : 0;
    Owned& previous_slot = has_previous ? palette.slot[previous_index] : current_slot;
    static std::atomic<int> s_trace_draws{ 0 };
    if (g_debug && comp == 0 && s_trace_draws.fetch_add(1) < 64)
        logger::write("PTSKIN-TRACE draw mod=%d comp=%d frame=%u out=%llx slot=%d (frame %u) previous=%d (frame %u) "
                      "pso=%llx list_type=%d",
            dm, comp, g_frame_id, (unsigned long long)section.out_va, current_index, palette.frame[current_index],
            previous_index, has_previous ? palette.frame[previous_index] : 0, (unsigned long long)state.gfx_pso,
            (int)d3d12_cl->GetType());
    LogPtSkinReadback();
    PtSkinGpu::Outputs* outputs = OutputsOf(device.Get(), once_key, vertices);
    if (!outputs) return {};
    const uint64_t gen = g_command_lists.Gen(cmd_list);
    Xit(d3d12_cl, gen, current_slot, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (has_previous) Xit(d3d12_cl, gen, previous_slot, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    ReadyModBufferForSkin(d3d12_cl, positions.res);
    ReadyModBufferForSkin(d3d12_cl, vectors.res);
    ReadyModBufferForSkin(d3d12_cl, blend.res);
    ReadyModBufferForSkin(d3d12_cl, colors.res);
    Xit(d3d12_cl, gen, outputs->positions, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Xit(d3d12_cl, gen, outputs->tangents, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Xit(d3d12_cl, gen, outputs->colors, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (has_previous) Xit(d3d12_cl, gen, outputs->previous_positions, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    uint32_t constants[4] = { vertices, bone_count, 0, 1 };
    memcpy(&constants[2], &section.scale, sizeof(float));
    d3d12_cl->SetComputeRootSignature(s_gpu.root_signature.Get());
    d3d12_cl->SetPipelineState(s_gpu.pso.Get());
    d3d12_cl->SetComputeRoot32BitConstants(0, 4, constants, 0);
    d3d12_cl->SetComputeRootShaderResourceView(1, current_slot.va());
    d3d12_cl->SetComputeRootShaderResourceView(2, positions.va);
    d3d12_cl->SetComputeRootShaderResourceView(3, vectors.va);
    d3d12_cl->SetComputeRootShaderResourceView(4, blend.va);
    d3d12_cl->SetComputeRootShaderResourceView(5, colors.va);
    d3d12_cl->SetComputeRootUnorderedAccessView(6, outputs->positions.va());
    d3d12_cl->SetComputeRootUnorderedAccessView(7, outputs->tangents.va());
    d3d12_cl->SetComputeRootUnorderedAccessView(8, outputs->colors.va());
    d3d12_cl->Dispatch((vertices + 63) / 64, 1, 1);
    if (has_previous) { // the same vertices with the previous frame's palette, positions only
        uint32_t previous_constants[4] = { vertices, previous_bone_count, 0, 0 };
        memcpy(&previous_constants[2], &section.scale, sizeof(float));
        d3d12_cl->SetComputeRoot32BitConstants(0, 4, previous_constants, 0);
        d3d12_cl->SetComputeRootShaderResourceView(1, previous_slot.va());
        d3d12_cl->SetComputeRootUnorderedAccessView(6, outputs->previous_positions.va());
        d3d12_cl->Dispatch((vertices + 63) / 64, 1, 1);
    }

    Xit(d3d12_cl, gen, outputs->tangents, kDrawReadState);
    Xit(d3d12_cl, gen, outputs->colors, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    if (g_debug && comp == s_gpu.readback_done % 8 && s_gpu.readback_frame == 0 && s_gpu.readback_done < 16 &&
        g_frame_id >= s_gpu.readback_next_frame) {
        if (!s_gpu.readback) {
            s_gpu.readback = WwmiMakeBuffer(device.Get(), 512, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST);
            void* mapped = nullptr;
            if (s_gpu.readback && SUCCEEDED(s_gpu.readback->Map(0, nullptr, &mapped)))
                s_gpu.readback_data = static_cast<const uint8_t*>(mapped);
        }
        if (s_gpu.readback_data) {
            Xit(d3d12_cl, gen, current_slot, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Xit(d3d12_cl, gen, outputs->positions, D3D12_RESOURCE_STATE_COPY_SOURCE);
            d3d12_cl->CopyBufferRegion(s_gpu.readback.Get(), 0, current_slot.res.Get(), 0, 96);
            for (uint32_t i = 0; i < 16; ++i) { // vertex i * vertices / 16
                const uint64_t vertex = (uint64_t)i * vertices / 16;
                d3d12_cl->CopyBufferRegion(s_gpu.readback.Get(), 96 + i * 12, outputs->positions.res.Get(),
                    vertex * kModPositionStride, kModPositionStride);
            }
            s_gpu.readback_vertices = vertices;
            s_gpu.readback_frame = g_frame_id;
            s_gpu.readback_key = once_key;
            s_gpu.readback_slot_frame = palette.frame[current_index];
        }
    }
    Xit(d3d12_cl, gen, outputs->positions, kDrawReadState);
    if (has_previous) Xit(d3d12_cl, gen, outputs->previous_positions, kDrawReadState);
    Xit(d3d12_cl, gen, current_slot, D3D12_RESOURCE_STATE_COMMON);
    if (has_previous) Xit(d3d12_cl, gen, previous_slot, D3D12_RESOURCE_STATE_COMMON);
    if (const uint64_t game_pso = state.gfx_restore())
        d3d12_cl->SetPipelineState(reinterpret_cast<ID3D12PipelineState*>(game_pso));
    const bool compute_restored = RestoreGameComputeRoot(cmd_list, d3d12_cl);

    // The game's layout: position, tangents, color, UVs, previous position.
    const uint64_t previous_va = has_previous ? outputs->previous_positions.va() : outputs->positions.va();
    const D3D12_VERTEX_BUFFER_VIEW views[5] = {
        { outputs->positions.va(), vertices * kModPositionStride, kModPositionStride },
        { outputs->tangents.va(), vertices * kOutTangentStride, kOutTangentStride },
        { outputs->colors.va(), vertices * kModColorStride, kModColorStride },
        { texcoords.va, texcoords.size, kModTexcoordStride },
        { previous_va, vertices * kModPositionStride, kModPositionStride },
    };
    d3d12_cl->IASetVertexBuffers(0, 5, views);

    // The pass-through VS takes only the position from the input layout. Tangents, UVs, the color and the previous
    // position it loads by vertex id from buffer SRVs, which the game points at its own mesh.
    ReadyModBufferForSkin(d3d12_cl, texcoords.res);
    PtSkinPass::FetchViews fetch;
    fetch.tan = outputs->tangents.res.Get();
    fetch.prev = has_previous ? outputs->previous_positions.res.Get() : outputs->positions.res.Get();
    fetch.uv = texcoords.res;
    fetch.uv_first = texcoords.res ? (texcoords.va - texcoords.res->GetGPUVirtualAddress()) / 4 : 0;
    fetch.uv_elems = texcoords.size / 4;
    fetch.color = colors.res;
    fetch.color_first = colors.res ? (colors.va - colors.res->GetGPUVirtualAddress()) / kModColorStride : 0;
    fetch.vcount = vertices;
    const VfRestore vf_restore = StampVertexFetch(cmd_list, d3d12_cl, fetch, dm, comp);
    s_ptskin_draws.fetch_add(1, std::memory_order_relaxed);
    if (LogOnce(14, once_key))
        logger::write("PTSKIN mod=%d comp=%d: %u vertices skinned with the section at %llx (%zu in the game's vb0; %u game vertices, %u bones, "
                      "scale %.3f, previous frame %s)%s",
            dm, comp, vertices, (unsigned long long)section.out_va, section_total, section.count, bone_count, section.scale,
            has_previous ? "skinned" : "not known yet", compute_restored ? "" : " -- no game compute root to put back");
    return vf_restore;
}

namespace {

/** Copies a layout's vertex-stage register maps out of the registry (LayoutRegistry::WithRegMap's callback). */
struct CopyVsRegMap {
    RegBind* srvs;
    RegBind* cbs;

    /**
     * @param map The layout's register map.
     */
    void operator()(const LayoutRegMap& map) const
    {
        memcpy(srvs, map.t_vs, sizeof(map.t_vs));
        memcpy(cbs, map.b_vs, sizeof(map.b_vs));
    }
};

} // namespace

/**
 * Points the buffer SRVs the pass-through vertex shader loads by vertex id at the mod's buffers, and gives the draw
 * a copy of the game's LocalVF constants (vs-b2) that indexes them from vertex 0 with four UV sets. Which register
 * holds what differs per shader, so each is told by the format of the view the game has bound there.
 *
 * @param cmd_list The game's command list (ReShade's).
 * @param cl       The same list.
 * @param fv       The mod's buffers.
 * @param dm       The mod, for the log.
 * @param comp     The component, for the log.
 * @return What to put back after the draw: the game's LocalVF and whether a shadow bind is needed.
 */
VfRestore RtDrawPass::StampVertexFetch(reshade::api::command_list* cmd_list, ID3D12GraphicsCommandList* cl,
    const PtSkinPass::FetchViews& fv, int dm, int comp)
{
    VfRestore restore;
    if (g_settings.ptskin_test == 2 || !g_addon.d3d12_device) return restore;
    const CmdState& state = g_command_lists.Gfx(cmd_list);
    const uint64_t layout = g_command_lists.GfxLayout(cmd_list);
    RegBind srvs[RegisterFileWidth(TargetClass::ShaderResource)];
    RegBind cbs[RegisterFileWidth(TargetClass::ConstantBuffer)];
    const CopyVsRegMap copy{ srvs, cbs };
    if (!g_layouts.WithRegMap(layout, copy)) return restore;
    const DxilSrvRegs reads = g_pipelines.SrvRegs(state.gfx_pso, 'v');
    reshade::api::device* device = cmd_list->get_device();
    const int log_key = (int)(((state.gfx_pso ^ (state.gfx_pso >> 32)) * 31u) & 0x7fffffff);
    const bool log = LogOnce(15, log_key);
    std::string report;

    std::lock_guard lock(fetch_mutex_);
    if (!fetch_heap_) {
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = kFetchRing;
        if (FAILED(g_addon.d3d12_device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&fetch_heap_)))) return restore;
    }
    const UINT increment = g_addon.d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    std::vector<ShadowSlotOvr> slots;
    const unsigned register_count = (unsigned)RegisterFileWidth(TargetClass::ShaderResource);
    for (unsigned reg = 0; reg < register_count && reg < 64; ++reg) {
        if (reads.parsed && !reads.has(reg)) continue;
        const RegBind& bind = srvs[reg];
        if (bind.param >= 64 || !bind.is_table) continue;
        const TableStart table = TableStartOf(device, state.root[bind.param].table);
        if (!table) continue;
        ResInfo info = {};
        if (!g_descriptors.Find({ table.heap, table.base + bind.offset }, info) || !info.is_buffer || !info.view_handle)
            continue;
        const reshade::api::resource_view_desc view = device->get_resource_view_desc({ info.view_handle });

        D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        ID3D12Resource* resource = nullptr;
        const char* role = nullptr;
        switch (view.format) {
        case reshade::api::format::r16g16b16a16_snorm: // two per vertex: TangentX, TangentZ
            resource = fv.tan;
            desc.Format = DXGI_FORMAT_R16G16B16A16_SNORM;
            desc.Buffer.NumElements = fv.vcount * 2;
            role = "tangents";
            break;
        case reshade::api::format::r16g16_float: // four UV sets per vertex
            resource = fv.uv;
            desc.Format = DXGI_FORMAT_R16G16_FLOAT;
            desc.Buffer.FirstElement = fv.uv_first;
            desc.Buffer.NumElements = fv.uv_elems;
            role = "UVs";
            break;
        case reshade::api::format::r32_float: // three per vertex: the previous position
            resource = fv.prev;
            desc.Format = DXGI_FORMAT_R32_FLOAT;
            desc.Buffer.NumElements = fv.vcount * 3;
            role = "previous positions";
            break;
        case reshade::api::format::r8g8b8a8_unorm: // the mod's colors are the game's bytes: read them the game's way
        case reshade::api::format::b8g8r8a8_unorm:
            resource = fv.color;
            desc.Format = (DXGI_FORMAT)view.format;
            desc.Buffer.FirstElement = fv.color_first;
            desc.Buffer.NumElements = fv.vcount;
            role = "colors";
            break;
        default: break;
        }
        if (log) {
            char part[96];
            snprintf(part, sizeof(part), " t%u=format %u (%s)", reg, (unsigned)view.format, role ? role : "left alone");
            report += part;
        }
        if (!resource) continue;
        if (fetch_next_ >= kFetchRing) fetch_next_ = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = fetch_heap_->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += (SIZE_T)fetch_next_++ * increment;
        g_addon.d3d12_device->CreateShaderResourceView(resource, &desc, handle);
        slots.push_back(ShadowSlotOvr{ bind.param, bind.offset, handle });
    }
    if (!slots.empty()) {
        g_rfx_bind.AddExtraSlots(slots.data(), (unsigned)slots.size());
        restore.fetch = true;
    }

    // LocalVF (vs-b2): int4 { color index mask, UV set count, light map UV index, first vertex }.
    int32_t local_vf[8] = {};
    const RegBind& cb = cbs[2];
    bool cb_copied = false;
    if (!slots.empty() && cb.param < 64 && !cb.is_table && state.root[cb.param].va &&
        g_game_cbs.ReadRootCbv(cmd_list, cb.param, local_vf, 24)) {
        int32_t ours[8];
        memcpy(ours, local_vf, sizeof(ours));
        ours[1] = 4; // the mod's vb2 holds four UV sets per vertex
        ours[3] = 0; // our buffers start at the mod's vertex 0
        if (lvf_ring_.Ensure(g_addon.d3d12_device, 256 * 1024)) {
            uint64_t va = 0;
            uint8_t* slot = lvf_ring_.Take(&va);
            if (slot) {
                memset(slot, 0, CbRing::kSlot);
                memcpy(slot, ours, 24);
                cl->SetGraphicsRootConstantBufferView(cb.param, va);
                restore.param = cb.param;
                restore.va = state.root[cb.param].va;
                cb_copied = true;
            }
        }
    }
    if (log)
        logger::write("PTSKIN-FETCH mod=%d comp=%d pso=%llx:%s | LocalVF (root %u%s) = %d %d %d %d %d %d -> %s", dm, comp,
            (unsigned long long)state.gfx_pso, report.empty() ? " no buffer SRV in the vertex stage" : report.c_str(),
            (unsigned)cb.param, cb.is_table ? ", in a table" : "", local_vf[0], local_vf[1], local_vf[2], local_vf[3],
            local_vf[4], local_vf[5], cb_copied ? "replaced with UV sets 4, first vertex 0" : "left as it is");
    return restore;
}

RtPalResult RtDrawPass::PalPublish(uint64_t, int, const WwmiMod::Component*, uint64_t)
{
    return {};
}

// ---- RTCSDUMP: the game's skin-cache compute shaders, saved for offline study ----

namespace {

// At most this many shaders are saved in one run.
constexpr uint32_t kCsDumpCap = 64;
std::atomic<uint32_t> s_cs_dumped{ 0 };

/**
 * Tells whether a shader's bytecode names a skin-cache entry point. Unreal's GPU skin cache shaders keep their
 * entry point name ("SkinCacheUpdateBatchCS", "...RecomputeTangents...") as plain text in the DXIL container.
 *
 * @param code      The shader bytecode.
 * @param code_size Its size in bytes.
 * @return true when the text "SkinCache" or "GpuSkin" or "RecomputeTangents" occurs in it.
 */
bool LooksLikeSkinCacheShader(const void* code, size_t code_size)
{
    const std::string_view bytes(static_cast<const char*>(code), code_size);
    return bytes.find("SkinCache") != std::string_view::npos || bytes.find("GpuSkin") != std::string_view::npos ||
           bytes.find("RecomputeTangents") != std::string_view::npos;
}

} // namespace

void RtNoteComputeShader(uint64_t cs_hash, const void* code, size_t code_size)
{
    if (!code || code_size == 0) return;
    if (!LooksLikeSkinCacheShader(code, code_size)) return;
    NoteSkinCbLayout(cs_hash, code, code_size);
    if (!g_debug) return;
    if (s_cs_dumped.fetch_add(1, std::memory_order_relaxed) >= kCsDumpCap) return;

    const std::wstring folder = AddonHomeDir() + L"wwmi_dx12\\rt_dump\\";
    CreateDirectoryW((AddonHomeDir() + L"wwmi_dx12").c_str(), nullptr);
    CreateDirectoryW(folder.c_str(), nullptr);
    wchar_t name[40];
    swprintf_s(name, L"%016llx.cso", (unsigned long long)cs_hash);
    const std::wstring path = folder + name;
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        logger::write("RTCSDUMP cs=%016llx: could not write %S (error %lu)", (unsigned long long)cs_hash, path.c_str(),
            GetLastError());
        return;
    }
    DWORD written = 0;
    WriteFile(file, code, (DWORD)code_size, &written, nullptr);
    CloseHandle(file);
    logger::write("RTCSDUMP cs=%016llx: %zu bytes saved to %S", (unsigned long long)cs_hash, code_size, path.c_str());
}

/**
 * Debug: saves the vertex shader of each pass-through PSO (the draws PTSKIN feeds) to rt_dump\vs_<hash>.cso, so
 * what it fetches by vertex id, outside its input layout, can be read from its disassembly.
 *
 * @param pso     The new graphics pipeline.
 * @param vs_hash Its vertex shader's hash.
 * @param vs_code The vertex shader's bytecode.
 * @param vs_size Its size in bytes.
 */
void RtNoteGraphicsPipeline(uint64_t pso, uint64_t vs_hash, const void* vs_code, size_t vs_size)
{
    {
        // A handle can be a destroyed PSO's: forget what IsPassThroughPso remembered for it.
        std::lock_guard lock(s_pso_mutex);
        s_pass_through.erase(pso);
    }
    if (!vs_code || vs_size == 0) return;
    char layout[1024] = {};
    MigotoPort::DescribeInputLayout(pso, layout, sizeof(layout));
    static std::atomic<int> s_calls{ 0 };
    if (s_calls.fetch_add(1) < 3)
        logger::write("RTVSDUMP probe: debug=%d vs=%016llx layout '%s'", (int)g_debug, (unsigned long long)vs_hash, layout);
    if (!g_debug || !strstr(layout, "ATTRIBUTE14@s4+0:f6")) return;
    static std::mutex s_mutex;
    static std::set<uint64_t> s_saved;
    {
        std::lock_guard lock(s_mutex);
        if (s_saved.size() >= 32 || !s_saved.insert(vs_hash).second) return;
    }
    const std::wstring folder = AddonHomeDir() + L"wwmi_dx12\\rt_dump\\";
    CreateDirectoryW((AddonHomeDir() + L"wwmi_dx12").c_str(), nullptr);
    CreateDirectoryW(folder.c_str(), nullptr);
    wchar_t name[48];
    swprintf_s(name, L"vs_%016llx.cso", (unsigned long long)vs_hash);
    const std::wstring path = folder + name;
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        logger::write("RTVSDUMP vs=%016llx: could not write %S (error %lu)", (unsigned long long)vs_hash, path.c_str(),
            GetLastError());
        return;
    }
    DWORD written = 0;
    WriteFile(file, vs_code, (DWORD)vs_size, &written, nullptr);
    CloseHandle(file);
    logger::write("RTVSDUMP vs=%016llx pso=%llx: pass-through vertex shader, %zu bytes saved to %S",
        (unsigned long long)vs_hash, (unsigned long long)pso, vs_size, path.c_str());
}

// ---- overlay and presence ----

void RtDrawOverlay()
{
    ImGui::TextDisabled("  ray tracing (portable RT module)");
    ImGui::TextDisabled("  BLAS builds: %llu   rebuilt (skinned): %llu   rings: %zu",
        (unsigned long long)s_builds_seen.load(), (unsigned long long)s_dynamic_builds.load(), g_rt_rings.Count());
    ImGui::TextDisabled("  PTSKIN draws: %llu   left as they are: %llu", (unsigned long long)s_ptskin_draws.load(),
        (unsigned long long)s_ptskin_skipped.load());
}

bool RtModulePresent()
{
    return true; // the mods stay on while the game ray traces, whatever the ini's rt_experimental says
}
