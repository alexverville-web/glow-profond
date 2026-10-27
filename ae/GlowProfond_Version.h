/* Shared between GlowProfond.cpp and the PiPL resource (GlowProfondPiPL.r). */
/* The PiPL and the code must agree on version and flags, or After Effects */
/* refuses to load the effect; GlowProfond.cpp static_asserts the flags. */
/* Include guards rather than #pragma once: Rez reads this file too. */
#ifndef GLOW_PROFOND_VERSION_H
#define GLOW_PROFOND_VERSION_H

#define GP_NAME "Glow Profond"
#define GP_MATCH_NAME "GMK GlowProfond"
#define GP_CATEGORY "Gimmick"
#define GP_SUPPORT_URL "https://gimmickstudio.ca"

#define GP_MAJOR_VERSION 0
#define GP_MINOR_VERSION 1
#define GP_BUG_VERSION 0
#define GP_STAGE_VERSION 0 /* PF_Stage_DEVELOP */
#define GP_BUILD_VERSION 1

/* PF_VERSION(major, minor, bug, stage, build), spelled out so Rez can use it. */
#define GP_VERSION_PACKED                                                            \
    ((GP_MAJOR_VERSION << 19) | (GP_MINOR_VERSION << 15) | (GP_BUG_VERSION << 11) | \
     (GP_STAGE_VERSION << 9) | GP_BUILD_VERSION)

/* PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_I_EXPAND_BUFFER */
#define GP_OUT_FLAGS 0x02000200
/* PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER | */
/* PF_OutFlag2_SUPPORTS_THREADED_RENDERING | PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG */
#define GP_OUT_FLAGS_2 0x08001408

#endif  /* GLOW_PROFOND_VERSION_H */
