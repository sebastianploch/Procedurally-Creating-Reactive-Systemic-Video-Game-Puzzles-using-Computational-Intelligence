#include "AssetEditor/EdNode_PSENode.h"
#include "AssetEditor/EdGraph_PSE.h"
#include "Kismet2/Kismet2NameValidators.h"
#include "Kismet2/BlueprintEditorUtils.h"

#define LOCTEXT_NAMESPACE "EdNode_PSENode"

UEdNode_PSENode::UEdNode_PSENode()
{
	bCanRenameNode = true;
}

void UEdNode_PSENode::SetNode(UPuzzleSequencerNode* InNode)
{
	Node = InNode;
}

UEdGraph_PSE* UEdNode_PSENode::GetEdGraph() const
{
	return Cast<UEdGraph_PSE>(GetGraph());
}

void UEdNode_PSENode::AllocateDefaultPins()
{
	CreatePin(EGPD_Input, "MultipleNodes", FName(), TEXT("In"));
	CreatePin(EGPD_Output, "MultipleNodes", FName(), TEXT("Out"));
}

FText UEdNode_PSENode::GetNodeTitle(ENodeTitleType::Type TitleType) const
{
	if (!IsValid(Node))
	{
		return Super::GetNodeTitle(TitleType);
	}
	else
	{
		return Node->GetNodeTitle();
	}
}

void UEdNode_PSENode::PrepareForCopying()
{
	Node->Rename(nullptr, this, REN_DontCreateRedirectors | REN_DoNotDirty);
}

void UEdNode_PSENode::AutowireNewNode(UEdGraphPin* FromPin)
{
	Super::AutowireNewNode(FromPin);

	if (FromPin)
	{
		if (GetSchema()->TryCreateConnection(FromPin, GetInputPin()))
		{
			FromPin->GetOwningNode()->NodeConnectionListChanged();
		}
	}
}

FLinearColor UEdNode_PSENode::GetBackgroundColour() const
{
	return Node == nullptr ? FLinearColor::Black : Node->GetBackgroundColor();
}

UEdGraphPin* UEdNode_PSENode::GetInputPin() const
{
	return Pins[0];
}

UEdGraphPin* UEdNode_PSENode::GetOutputPin() const
{
	return Pins[1];
}

#if WITH_EDITOR
void UEdNode_PSENode::PostEditUndo()
{
	UEdGraphNode::PostEditUndo();
}
#endif

#undef LOCTEXT_NAMESPACE
