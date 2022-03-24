#pragma once

#include "CoreMinimal.h"
#include "EdGraph/EdGraphNode.h"
#include "EdNode_PSEEdge.generated.h"

class UPuzzleSequencerNode;
class UPuzzleSequencerEdge;
class UEdNode_PSENode;

UCLASS(MinimalAPI)
class UEdNode_PSEEdge : public UEdGraphNode
{
	GENERATED_BODY()

public:
	UEdNode_PSEEdge();

	UPROPERTY()
	class UEdGraph* Graph{nullptr};

	UPROPERTY(VisibleAnywhere, Instanced, Category="PuzzleSequencer")
	UPuzzleSequencerEdge* Edge{nullptr};

	void SetEdge(UPuzzleSequencerEdge* InEdge);

	virtual void AllocateDefaultPins() override;
	virtual FText GetNodeTitle(ENodeTitleType::Type TitleType) const override;
	virtual void PinConnectionListChanged(UEdGraphPin* Pin) override;
	virtual void PrepareForCopying() override;

	virtual UEdGraphPin* GetInputPin() const { return Pins[0]; }
	virtual UEdGraphPin* GetOutputPin() const { return Pins[1]; }

	void CreateConnections(UEdNode_PSENode* InStart, UEdNode_PSENode* InEnd);

	UEdNode_PSENode* GetStartNode();
	UEdNode_PSENode* GetEndNode();
};
