#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "GlowProfond_Version.h"

#ifndef AE_OS_WIN
	#include "AE_General.r"
#endif

resource 'PiPL' (16000) {
	{
		Kind {
			AEEffect
		},
		Name {
			GP_NAME
		},
		Category {
			GP_CATEGORY
		},
#ifdef AE_OS_WIN
		CodeWin64X86 {"EffectMain"},
#else
		CodeMacIntel64 {"EffectMain"},
		CodeMacARM64 {"EffectMain"},
#endif
		AE_PiPL_Version {
			2,
			0
		},
		AE_Effect_Spec_Version {
			PF_PLUG_IN_VERSION,
			PF_PLUG_IN_SUBVERS
		},
		AE_Effect_Version {
			GP_VERSION_PACKED
		},
		AE_Effect_Info_Flags {
			0
		},
		AE_Effect_Global_OutFlags {
			GP_OUT_FLAGS
		},
		AE_Effect_Global_OutFlags_2 {
			GP_OUT_FLAGS_2
		},
		AE_Effect_Match_Name {
			GP_MATCH_NAME
		},
		AE_Reserved_Info {
			0
		},
		AE_Effect_Support_URL {
			GP_SUPPORT_URL
		}
	}
};
