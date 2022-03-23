#include "PSEditor.h"
#include "PSENodeFactory.h"
#include "AssetTypeActions_PSE.h"
#include "AssetEditor/PSEStyle.h"

#define LOCTEXT_NAMESPACE "Editor_PuzzleSequencer"

void FPSEditor::StartupModule()
{
	FPSEStyle::Initialise();

	GraphPanelNodeFactory = MakeShareable(new FPSENodeFactory());
	FEdGraphUtilities::RegisterVisualNodeFactory(GraphPanelNodeFactory);

	IAssetTools& AssetTools = FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools").Get();

	PuzzleSequencerAssetCategoryBit = AssetTools.RegisterAdvancedAssetCategory(FName(TEXT("GenericGraph")), LOCTEXT("GenericGraphAssetCategory", "GenericGraph"));

	RegisterAssetTypeAction(AssetTools, MakeShareable(new FAssetTypeActions_PSE(PuzzleSequencerAssetCategoryBit)));
}

void FPSEditor::ShutdownModule()
{
	// Unregister all the asset types that we registered
	if (FModuleManager::Get().IsModuleLoaded("AssetTools"))
	{
		IAssetTools& AssetTools = FModuleManager::GetModuleChecked<FAssetToolsModule>("AssetTools").Get();
		for (int32 Index = 0; Index < CreatedAssetTypeActions.Num(); ++Index)
		{
			AssetTools.UnregisterAssetTypeActions(CreatedAssetTypeActions[Index].ToSharedRef());
		}
	}

	if (GraphPanelNodeFactory.IsValid())
	{
		FEdGraphUtilities::UnregisterVisualNodeFactory(GraphPanelNodeFactory);
		GraphPanelNodeFactory.Reset();
	}

	FPSEStyle::Shutdown();
}

void FPSEditor::RegisterAssetTypeAction(IAssetTools& InAssetTools, TSharedRef<IAssetTypeActions> InAction)
{
	InAssetTools.RegisterAssetTypeActions(InAction);
	CreatedAssetTypeActions.Add(InAction);
}

#undef LOCTEXT_NAMESPACE
