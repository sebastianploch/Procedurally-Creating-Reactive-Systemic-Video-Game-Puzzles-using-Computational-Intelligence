#include "AssetEditor/AssetEditorToolbar_PSE.h"
#include "AssetEditor/AssetEditor_PSE.h"
#include "AssetEditor/EditorCommands_PSE.h"

#define LOCTEXT_NAMESPACE "AssetEditorToolbar_PSE"

FAssetEditorToolbar_PSE::FAssetEditorToolbar_PSE(TSharedPtr<FAssetEditor_PSE> InPuzzleSequencerEditor)
	: Editor(InPuzzleSequencerEditor)
{
}

void FAssetEditorToolbar_PSE::AddPuzzleSequencerToolbar(TSharedPtr<FExtender> InExtender)
{
	check(Editor.IsValid());
	const TSharedPtr<FAssetEditor_PSE> editor = Editor.Pin();

	InExtender->AddToolBarExtension("Asset", EExtensionHook::After, editor->GetToolkitCommands(), FToolBarExtensionDelegate::CreateSP(this, &FAssetEditorToolbar_PSE::FillPuzzleSequencerGraphToolbar));
	editor->AddToolbarExtender(InExtender);
}

void FAssetEditorToolbar_PSE::FillPuzzleSequencerGraphToolbar(FToolBarBuilder& InToolBarBuilder)
{
	check(Editor.IsValid())
	TSharedPtr<FAssetEditor_PSE> editor = Editor.Pin();

	InToolBarBuilder.BeginSection("Puzzle Sequencer");
	{
		InToolBarBuilder.AddToolBarButton(FEditorCommands_PSE::Get().GraphSettings,
		                                  NAME_None,
		                                  LOCTEXT("GraphSettings_Label", "Graph Settings"),
		                                  LOCTEXT("GraphSettings_ToolTip", "Show the Graph Settings"),
		                                  FSlateIcon(FEditorStyle::GetStyleSetName(), "LevelEditor.GameSettings"));
	}
	InToolBarBuilder.EndSection();
}

#undef LOCTEXT_NAMESPACE
