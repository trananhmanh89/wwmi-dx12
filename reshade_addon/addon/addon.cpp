#include "addon.h"
#include "home_dir.h"

#include <windows.h>

Addon g_addon;

std::wstring AddonHomeDir()
{
    // Ask Windows which module contains this very function: that is wwmi_dx12.addon64,
    // wherever ReShade loaded it from.
    HMODULE self = nullptr;
    const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&AddonHomeDir), &self) || !self) {
        return std::wstring();
    }

    wchar_t path[MAX_PATH] = {};
    if (!GetModuleFileNameW(self, path, MAX_PATH)) {
        return std::wstring();
    }

    std::wstring folder = path;
    const size_t last_slash = folder.find_last_of(L"\\/");
    if (last_slash == std::wstring::npos) {
        return std::wstring();
    }
    return folder.substr(0, last_slash + 1);
}
