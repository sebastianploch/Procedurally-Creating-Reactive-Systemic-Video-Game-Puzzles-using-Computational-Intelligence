#include "PSEditor.h"
#include "PSENodeFactory.h"
#include "AssetTypeActions_PSE.h"
#include "AssetEditor/PSEStyle.h"

#define LOCTEXT_NAMESPACE "Editor_PuzzleSequencer"

void FPSEditor::StartupModule()
{
}

void FPSEditor::ShutdownModule()
{
}

void FPSEditor::RegisterAssetTypeAction(IAssetTools& InAssetTools, TSharedRef<IAssetTypeActions> InAction)
{
	InAssetTools.RegisterAssetTypeActions(InAction);
	CreatedAssetTypeActions.Add(InAction);
}

#undef LOCTEXT_NAMESPACE
