#pragma once

#include "CoreMinimal.h"
#include "Settings_PSE.h"
#include "PuzzleSequencer.h"

class PUZZLESEQUENCEREDITOR_API FAssetEditor_PSE : public FAssetEditorToolkit, public FNotifyHook, public FGCObject
{
public:
	FAssetEditor_PSE();
	virtual ~FAssetEditor_PSE() override = default;

	void InitPuzzleSequencerAssetEditor(const EToolkitMode::Type InMode, const TSharedPtr<IToolkitHost>& InInitToolkitHost, UPuzzleSequencer* InGraph);


	virtual void RegisterTabSpawners(const TSharedRef<FTabManager>& TabManager) override;
	virtual void UnregisterTabSpawners(const TSharedRef<FTabManager>& TabManager) override;

	virtual FName GetToolkitFName() const override;
	virtual FText GetBaseToolkitName() const override;
	virtual FText GetToolkitName() const override;
	virtual FText GetToolkitToolTipText() const override;
	virtual FLinearColor GetWorldCentricTabColorScale() const override;
	virtual FString GetWorldCentricTabPrefix() const override;
	virtual FString GetDocumentationLink() const override;
	virtual void SaveAsset_Execute() override;

	void UpdateToolbar();
	//TSharedPtr<class FAssetEd>
};
