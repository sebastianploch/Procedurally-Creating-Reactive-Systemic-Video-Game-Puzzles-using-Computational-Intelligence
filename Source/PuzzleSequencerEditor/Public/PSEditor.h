#pragma once
#include "Modules/ModuleManager.h"
#include "PSEModule.h"
#include <IAssetTools.h>
#include <EdGraphUtilities.h>

class PUZZLESEQUENCEREDITOR_API FPSEditor : public IPuzzleSequencerEditor
{
private:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	void RegisterAssetTypeAction(IAssetTools& InAssetTools, TSharedRef<IAssetTypeActions> InAction);

private:
	TArray<TSharedPtr<IAssetTypeActions>> CreatedAssetTypeActions{};
	EAssetTypeCategories::Type PuzzleSequencerAssetCategoryBit{EAssetTypeCategories::None};
	TSharedPtr<FGraphPanelNodeFactory> GraphPanelNodeFactory{};
};

IMPLEMENT_MODULE(FPSEditor, PuzzleSequencerEditor)
