#pragma once

#include "Modules/ModuleManager.h"

class IPuzzleSequencer : public IModuleInterface
{
public:
	static IPuzzleSequencer& Get()
	{
		return FModuleManager::LoadModuleChecked<IPuzzleSequencer>("PuzzleSequencer");
	}

	static bool IsAvailable()
	{
		return FModuleManager::Get().IsModuleLoaded("PuzzleSequencer");
	}
};
