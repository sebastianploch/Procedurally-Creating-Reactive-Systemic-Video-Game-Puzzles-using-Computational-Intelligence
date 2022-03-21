#pragma once

#include "CoreMinimal.h"

class FAssetEditor_PSE;
class FExtender;
class FToolBarBuilder;

class PUZZLESEQUENCEREDITOR_API FAssetEditorToolbar_PSE : public TSharedFromThis<FAssetEditorToolbar_PSE>
{
public:
	FAssetEditorToolbar_PSE(TSharedPtr<FAssetEditor_PSE> InPuzzleSequencerEditor);

	void AddPuzzleSequencerToolbar(TSharedPtr<FExtender> InExtender);

private:
	void FillPuzzleSequencerGraphToolbar(FToolBarBuilder& InToolBarBuilder);

protected:
	TWeakPtr<FAssetEditor_PSE> Editor{};
};
