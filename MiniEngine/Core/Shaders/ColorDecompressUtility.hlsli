//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author(s):  James Stanard
//

#ifndef _XBOX_ONE
#error "This shader utility is only usable on Xbox"
#endif

// Returns a 4-bit ClearMask value for an 8x8 tile.  The order of the bits for
// the 4x4 sub-tiles are:
//
//   0001 = Upper-Left
//   0010 = Upper-Right
//   0100 = Lower-Left
//   1000 = Lower-Right
//
// If a bit is set, that sub-tile is NOT clear.  This function assumes the tiling pattern
// used with non-MSAA render targets.
//
// Only P4_16x16 (Durango) and P8_32x32_16x16 (Scorpio) are currently supported.  Each is indicated
// by the pipe count (4 or 8) in the CMaskInfo header.
uint GetCMask(ByteAddressBuffer CMaskBuffer, uint2 TileCoord, uint CMaskInfo)
{
    // Unpack the CMaskInfo descriptor passed in by the application
    const uint2 NumTiles = uint2(CMaskInfo, CMaskInfo >> 12) & 0xFFF;
    const uint PipeCount = (CMaskInfo >> 24) & 0x7F;
    const bool LinearAddressing = (CMaskInfo >> 31) == 1;

    // Dimensions of the macro tile for non-linear mode.  (In units of tiles, not pixels.)
    uint macroTileWidth = 8 * PipeCount;
    uint macroTileHeight = 32;

    uint tileY0 = TileCoord.y & 1;
    uint tileX1 = (TileCoord.x >> 1) & 1;
    uint elemIdx = (tileX1 ^ tileY0) | tileX1 << 1;
    uint elemIdxBits = 2;

    uint pipeMask = PipeCount - 1;
    uint pipe = (TileCoord.x ^ TileCoord.y ^ tileX1) & pipeMask;
    uint pipeBits = countbits(pipeMask);
    uint microRightShift = elemIdxBits + pipeBits - 4;

    // tilesPerPipe = macroTileWidth * macroTileHeight / PipeCount
    const uint tilesPerPipe = 256;

    const uint macroTileCountX = NumTiles.x / macroTileWidth; // clPitch
    const uint macroTileCountY = NumTiles.y / macroTileHeight;

    const uint slicePitch = NumTiles.x * NumTiles.y / PipeCount;

    // for 2D array and 3D textures (including cube maps)
    uint tileSlice = 0; // tileZ
    uint sliceOffset = slicePitch * tileSlice;

    // macro tile location
    uint macroX = TileCoord.x / macroTileWidth;
    uint macroY = TileCoord.y / macroTileHeight;
    uint macroOffset = LinearAddressing ? 0 : (macroX + macroTileCountX * macroY) * tilesPerPipe;

    // micro (4x4 tile) tiling
    uint microX = (LinearAddressing ? TileCoord.x : (TileCoord.x % macroTileWidth)) / 4;
    uint microY = (LinearAddressing ? TileCoord.y : (TileCoord.y % macroTileHeight)) / 4;
    uint microPitch = (LinearAddressing ? NumTiles.x : macroTileWidth) / 4;
    uint microOffset = ((microX + microY * microPitch) >> microRightShift) << elemIdxBits | elemIdx;

    uint tileIndex = sliceOffset + macroOffset + microOffset;

    // Squeeze the 2-bit pipe value into the address.
    uint nibbleAddress = (tileIndex & ~0x1FF) << pipeBits | pipe << 9 | (tileIndex & 0x1FF);

    // Convert the nibble address to the address of the word and the bit offset
    // into the word.
    uint wordAddress = (nibbleAddress / 8) * 4;	// 4 bytes per word
    uint bitOffset = (nibbleAddress % 8) * 4;	// 4 bits per nibble

    return (CMaskBuffer.Load(wordAddress) >> bitOffset) & 0xF;
}
