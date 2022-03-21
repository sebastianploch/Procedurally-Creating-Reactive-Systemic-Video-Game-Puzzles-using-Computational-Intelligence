#include "AssetEditor/EditorCommands_PSE.h"

#define LOCTEXT_NAMESPACE "EditorCommands_PuzzleSequencer"

FEditorCommands_PSE::FEditorCommands_PSE()
	: TCommands<FEditorCommands_PSE>("PuzzleSequencerEditor", NSLOCTEXT("Contexts", "PuzzleSequencerEditor", "Puzzle Sequencer Editor"), NAME_None, FEditorStyle::GetStyleSetName())
{
}

void FEditorCommands_PSE::RegisterCommands()
{
	UI_COMMAND(GraphSettings, "Graph Settings", "Graph Settings", EUserInterfaceActionType::Button, FInputChord());
}

#undef LOCTEXT_NAMESPACE
