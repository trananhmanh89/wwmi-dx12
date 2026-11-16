#pragma once
// matcap_patch.h -- OURS. The DX12 form of the KatFXMatcap mod's [ShaderRegex\zKatFXMatcap\Matcap] patch.
//
// The mod binds a matcap texture at ps-t125 (layer 1) and / or ps-t124 (layer 2), optional masks at ps-t126 /
// ps-t127, and its parameters in x125..w125 / x126..w126. On DX11 a regex rewrites the character pixel shader
// to blend them in. Here the pixel shader's DXIL is patched instead (matcap_dxil.inc, through rfx_dxil.cpp)
// and the draw runs on a PSO variant built with it. The patch is independent of RabbitFX's: it is layered over
// whatever the draw already uses -- the game's shader, the decensored one, or RabbitFX's patched one.
#include <reshade.hpp>
#include <d3d12.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../rabbitfx/pso_clone.h"
#include "../rabbitfx/rabbitfx_bind.h"

constexpr unsigned kMatcapReg = 53; // ps-t53 / t54: layer 1 / 2; t55 / t56: their masks. Ours: the ini's t124..t127 are
constexpr unsigned kMatcapParamsReg = 57; // outside the game's t0..t63 table. ps-t57: two float4, x125..w125 and x126..w126
constexpr unsigned kMatcapLayer1Reg = 125; // the ini's registers
constexpr unsigned kMatcapLayer2Reg = 124;
constexpr unsigned kMatcapMask1Reg = 126;
constexpr unsigned kMatcapMask2Reg = 127;
constexpr unsigned kMatcapParams1 = 125; // IniParams texels
constexpr unsigned kMatcapParams2 = 126;
constexpr const wchar_t* kMatcapNamespace = L"zKatFXMatcap"; // [ShaderRegex\zKatFXMatcap\Matcap]
constexpr const wchar_t* kMatcapRegexGroup = L"Matcap";

// What the shader the matcap is layered over is.
enum class MatcapBase : unsigned {
    Game = 0, // the PSO's own pixel shader
    Decensored = 1,
    Rfx = 2, // RabbitFX's patched one
    RfxDecensored = 3,
};

// A draw's matcap: the PSO to draw with and the descriptors to bind besides the game's own.
struct MatcapPlan {
    ID3D12PipelineState* pso = nullptr; // null: draw as the game does
    std::vector<WclSlotOvr> binds;
    unsigned ps_max_reg = 0; // the variant's highest t-register
    unsigned top_reg = 0; // the highest the bind needs
};

// The matcap variants of the game's PSOs, built off the drawing thread.
class MatcapWorker {
public:
    // `src_pso`'s variant over `base_code` (null: the PSO's own shader); nullptr while it is queued or when the
    // patch does not apply to this shader. `kind` is the layering, part of the variant's identity.
    ID3D12PipelineState* Variant(reshade::api::device* dev, uint64_t src_pso, MatcapBase kind, CodeBlob base_code,
        unsigned* ps_max_reg);
    void Stop();
    // The source's variants, queued jobs and settled answers leave the tables; the caller destroys the variants.
    void ForgetSource(uint64_t src_pso, std::vector<ID3D12PipelineState*>& dead);

private:
    struct Made {
        ID3D12PipelineState* pso; // nullptr = a settled no
        unsigned max_reg;
    };
    struct Job {
        reshade::api::device* dev;
        uint64_t src_pso;
        uint64_t key;
        CodeBlob base_code;
        PsoClone clone;
    };
    struct Thread {
        std::thread t;
        ~Thread()
        {
            if (t.joinable()) t.detach();
        }
    };
    void Build(Job& job);
    void Run();
    void Wake();

    std::mutex mutex_; // variants_, pending_, queue_, miss_logged_
    std::unordered_map<uint64_t, Made> variants_; // src | kind << 60
    std::unordered_set<uint64_t> pending_;
    std::unordered_set<uint64_t> miss_logged_;
    std::deque<Job> queue_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{ false };
    Thread worker_;
    std::mutex thread_mutex_;
};
extern MatcapWorker g_matcap;

// The matcap of this section draw: the textures the mod bound in this section and its parameters. `rfx_variant_in_use`
// and `decensor_on` say which shader the draw runs on without it. Returns an empty plan when the group is not loaded, the mod bound no matcap
// texture, the variant is not built yet, or the draw cannot take the extra descriptors.
MatcapPlan MatcapPlanDraw(reshade::api::command_list* cmd_list, uint64_t src_pso, bool rfx_variant_in_use,
    bool decensor_on, uint32_t srv_gen0, int mod, int comp);
