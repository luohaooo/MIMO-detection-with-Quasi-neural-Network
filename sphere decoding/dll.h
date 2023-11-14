#ifndef _DLL_H_
#define _DLL_H_

#if BUILDING_DLL
#define DLLIMPORT __declspec(dllexport)
#else
#define DLLIMPORT __declspec(dllimport)
#endif

DLLIMPORT void MIMO_mldDec(float* vfllrs, float* vfCxR_list, float* vfCxH_list, int LyrN, int SCnum, int ModuType, float maxL0);

#endif
