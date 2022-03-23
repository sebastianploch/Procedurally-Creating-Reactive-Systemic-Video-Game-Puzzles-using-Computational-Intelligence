#include "IPuzzleSequencer.h"

class FPuzzleSequencerModule : public IPuzzleSequencer
{
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};

IMPLEMENT_MODULE(FPuzzleSequencerModule, PuzzleSequencer)

void FPuzzleSequencerModule::StartupModule()
{
}

void FPuzzleSequencerModule::ShutdownModule()
{
}
