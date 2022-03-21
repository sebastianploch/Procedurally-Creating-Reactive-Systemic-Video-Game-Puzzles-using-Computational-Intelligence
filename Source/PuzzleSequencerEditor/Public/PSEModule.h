#pragma once
#include "Modules/ModuleManager.h"

class PUZZLESEQUENCEREDITOR_API IPuzzleSequencerEditor : public IModuleInterface
{
public:
	FORCEINLINE static IPuzzleSequencerEditor& Get()
	{
		return FModuleManager::LoadModuleChecked<IPuzzleSequencerEditor>("PuzzleSequencerEditor");
	}

	FORCEINLINE static bool IsAvailable()
	{
		return FModuleManager::Get().IsModuleLoaded("PuzzleSequencerEditor");
	}
};
