#pragma once

#include "CoreMinimal.h"
#include "EdGraph/EdGraphNode.h"
#include "PuzzleSequencerNode.h"
#include "EdNode_PSENode.generated.h"

class UEdNode_PSEEdge;
class UEdGraph_PSE;
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
	UEdGraph_PSE* GetEdGraph() const;

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
