#include "AssetTypeActions_PSE.h"
#include "PuzzleSequencer.h"
#include "AssetEditor/AssetEditor_PSE.h"

#define LOCTEXT_NAMESPACE "AssetTypeActions_PSE"

FAssetTypeActions_PSE::FAssetTypeActions_PSE(EAssetTypeCategories::Type InAssetCategory)
	: AssetCategory(InAssetCategory)
{
}

FText FAssetTypeActions_PSE::GetName() const
{
	return LOCTEXT("FPuzzleSequencerAssetTypeActionsName", "Puzzle Sequencer");
}

FColor FAssetTypeActions_PSE::GetTypeColor() const
{
	return FColor(129, 196, 115);
}

UClass* FAssetTypeActions_PSE::GetSupportedClass() const
{
	return UPuzzleSequencer::StaticClass();
}

void FAssetTypeActions_PSE::OpenAssetEditor(const TArray<UObject*>& InObjects, TSharedPtr<IToolkitHost> EditWithinLevelEditor)
{
	const EToolkitMode::Type mode = EditWithinLevelEditor.IsValid() ? EToolkitMode::WorldCentric : EToolkitMode::Standalone;
	for (const auto& object : InObjects)
	{
		if (UPuzzleSequencer* graph = Cast<UPuzzleSequencer>(object))
		{
			// TODO: Finish this duh
			//TShaderRef<FAssetEditor_PSE> newGraphEditor(new FAssetEditor_PSE());
		}
	}
}

uint32 FAssetTypeActions_PSE::GetCategories()
{
	return AssetCategory;
}

#undef LOCTEXT_NAMESPACE
