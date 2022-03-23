#pragma once

#include "CoreMinimal.h"
#include "EdGraph/EdGraph.h"
#include "EdGraph_PSE.generated.h"

class UPuzzleSequencer;
class UPuzzleSequencerNode;
class UPuzzleSequencerEdge;
class UEdNode_PSENode;
class UEdNode_PSEEdge;

UCLASS()
class PUZZLESEQUENCEREDITOR_API UEdGraph_PSE : public UEdGraph
{
	GENERATED_BODY()

public:
	virtual void RebuildGraph();

	UPuzzleSequencer* GetGraph();

	virtual bool Modify(bool bAlwaysMarkDirty = true) override;
	virtual void PostEditUndo() override;

	UPROPERTY(Transient)
	TMap<UPuzzleSequencerNode*, UEdNode_PSENode*> NodeMap{};

	UPROPERTY(Transient)
	TMap<UPuzzleSequencerEdge*, UEdNode_PSEEdge*> EdgeMap{};

protected:
	void Clear();

	void SortNodes(UPuzzleSequencerNode* InRootNode);
};
