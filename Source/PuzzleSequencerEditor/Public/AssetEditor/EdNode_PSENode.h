#pragma once

#include "CoreMinimal.h"
#include "EdGraph/EdGraphNode.h"
#include "PuzzleSequencerNode.h"
#include "SEdNode_PSENode.h"
#include "EdNode_PSENode.generated.h"

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
