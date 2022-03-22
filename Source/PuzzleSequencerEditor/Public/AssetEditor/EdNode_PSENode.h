#pragma once

#include "CoreMinimal.h"
#include "EdGraph/EdGraphNode.h"
#include "EdNode_PSENode.generated.h"

class UPuzzleSequencerNode;
class UEdNode_PSEEdge;
class SEdNode_PSENode;

UCLASS(MinimalAPI)
class UEdNode_PSENode : public UEdGraphNode
{
	GENERATED_BODY()

public:
	UEdNode_PSENode();

	UPROPERTY(VisibleAnywhere, Instanced, Category="PuzzleSequencer")
	UPuzzleSequencerNode* Node{nullptr};

	void SetNode(UPuzzleSequencerNode* InNode);
	// TODO: 	UEdGraph_GenericGraph* GetGenericGraphEdGraph();

	SEdNode_PSENode* SEdNode{nullptr};
	
	virtual void AllocateDefaultPins() override;
	virtual FText GetNodeTitle(ENodeTitleType::Type TitleType) const override;
	virtual void PrepareForCopying() override;
	virtual void AutowireNewNode(UEdGraphPin* FromPin) override;

	virtual FLinearColor GetBackgroundColour() const;
	virtual UEdGraphPin* GetInputPin() const;
	virtual UEdGraphPin* GetOutputPin() const;

#pragma region Editor
#if WITH_EDITOR
	virtual void PostEditUndo() override;
#endif
#pragma endregion Editor
};
