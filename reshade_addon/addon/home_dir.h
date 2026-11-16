#pragma once
#include <string>

/**
 * Returns the folder that holds wwmi_dx12.addon64, with a trailing backslash.
 *
 * The portable build keeps everything it writes here (wwmi_dx12.ini, logs, caches,
 * crash reports), so nothing is written into the game's folder.
 *
 * @return The add-on's folder, for example L"J:\\XXMI-Launcher-Portable\\WWMI-DX12\\",
 *         or an empty string if Windows cannot tell where the add-on was loaded from.
 */
std::wstring AddonHomeDir();
