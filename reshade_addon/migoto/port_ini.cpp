#include "CommandList.h"
#include "input.h"
#include "port_keygate.h"
#include "port_api.h"

void PortOptimiseCommandLists(HackerDevice* device);
void PortRegisterKeyBinding(const wchar_t* section, LPCWSTR ini_key, const wchar_t* key_name,
    shared_ptr<InputListener> listener, int auto_repeat, int down_delay, int up_delay,
    InputDisableScope input_disable_scope = InputDisableScope::INVALID);

// Runs just before the engine's LogInfo calls. LoadConfigFile logs "[Include]" after it has parsed d3dx.ini
// and before it reads the mods d3dx.ini includes, which is where PortRedirectModsFolder steps in.
void PortBeforeEngineLog(const char* fmt);
#include "log.h"
#undef LogInfo
#define LogInfo(fmt, ...)                                                                \
    do {                                                                                 \
        PortBeforeEngineLog(fmt);                                                        \
        if (gLogVerbosity >= LogVerbosity::INFO) fprintf(LogFile, fmt, __VA_ARGS__);     \
    } while (0)

#define optimise_command_lists PortOptimiseCommandLists
#define RegisterKeyBinding(...) PortRegisterKeyBinding(id, __VA_ARGS__)

#include "IniHandler.cpp"

#undef optimise_command_lists
#undef RegisterKeyBinding

namespace {

/**
 * Points d3dx.ini's `include_recursive = Mods` at the folder PortApi::SetModsFolder chose, and the saved
 * toggles at PortApi::ModsUserConfig, in the parsed copy only: d3dx.ini itself, which DX11 shares, is left
 * as it is.
 */
void PortRedirectModsFolder()
{
    const wchar_t* folder = PortApi::ModsFolder();
    if (!folder || !folder[0]) return;
    IniSections::iterator include = ini_sections.find(L"Include");
    if (include == ini_sections.end()) return;
    include->second.kv_map[L"user_config"] = PortApi::ModsUserConfig();
    for (IniLine& line : include->second.kv_vec) {
        if (_wcsicmp(line.first.c_str(), L"include_recursive") != 0) continue;
        wstring value = line.second;
        while (!value.empty() && (value.back() == L'\\' || value.back() == L'/')) value.pop_back();
        if (_wcsicmp(value.c_str(), L"Mods") != 0) continue;
        line.second = folder;
        line.raw_line = L"include_recursive = " + wstring(folder);
    }
}

} // namespace

void PortBeforeEngineLog(const char* fmt)
{
    if (fmt && fmt[0] == '[' && strcmp(fmt, "[Include]\n") == 0) PortRedirectModsFolder();
}

void PortOptimiseCommandLists(HackerDevice* device)
{
    MigotoPort::WrapObservedCommands(registered_command_lists);
    optimise_command_lists(device);
}

void PortRegisterKeyBinding(const wchar_t* section, LPCWSTR ini_key, const wchar_t* key_name,
    shared_ptr<InputListener> listener, int auto_repeat, int down_delay, int up_delay,
    InputDisableScope input_disable_scope)
{
    if (listener && section && !wcscmp(ini_key, L"Key")) {
        wstring ns;
        get_section_namespace(section, &ns);
        listener = MigotoPort::GateKeyListener(listener, ns, section);
    }
    RegisterKeyBinding(ini_key, key_name, listener, auto_repeat, down_delay, up_delay, input_disable_scope);
}

bool PortSectionFolder(const wchar_t* section, wstring* ret)
{
    wchar_t base[MAX_PATH];
    wstring rel;
    if (!get_namespaced_section_path(section, &rel))
        return false;
    MigotoPort::MigotoDir(base, MAX_PATH);
    *ret = wstring(base) + rel;
    while (!ret->empty() && (ret->back() == L'\\' || ret->back() == L'/'))
        ret->pop_back();
    return !ret->empty();
}

bool PortIniNamespace(const wchar_t* abs_ini_path, wstring* ret)
{
    wchar_t base[MAX_PATH];
    MigotoPort::MigotoDir(base, MAX_PATH);
    wstring path(abs_ini_path);
    wstring root(base);
    for (auto& ch : path)
        if (ch == L'/') ch = L'\\';
    for (auto& ch : root)
        if (ch == L'/') ch = L'\\';
    if (root.empty() || path.size() <= root.size() ||
        _wcsnicmp(path.c_str(), root.c_str(), root.size()) != 0)
        return false;
    const wstring rel = path.substr(root.size());

    for (auto& kv : ini_sections) {
        const IniSection& s = kv.second;
        const wstring&    file = s.ini_path.empty() ? s.ini_namespace : s.ini_path;
        if (_wcsicmp(file.c_str(), rel.c_str()) != 0)
            continue;
        *ret = s.ini_namespace;
        return !ret->empty();
    }
    return false;
}

void PortForEachIniFile(void (*cb)(const wstring& rel_path, const wstring& ns, void* user), void* user)
{
    std::set<wstring, WStringInsensitiveLess> seen;
    for (auto& kv : ini_sections) {
        const IniSection& s = kv.second;
        const wstring&    file = s.ini_path.empty() ? s.ini_namespace : s.ini_path;
        if (file.empty() || !seen.insert(file).second)
            continue;
        cb(file, s.ini_namespace, user);
    }
}
