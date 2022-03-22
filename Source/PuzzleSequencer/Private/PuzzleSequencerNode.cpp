#include "PuzzleSequencerNode.h"
#include "PuzzleSequencer.h"

#define LOCTEXT_NAMESPACE "PuzzleSequencerNode"

UPuzzleSequencerNode::UPuzzleSequencerNode()
{
#if WITH_EDITORONLY_DATA
	CompatibleGraphType = UPuzzleSequencer::StaticClass();

	BackgroundColor = FLinearColor::Black;
#endif
}

UPuzzleSequencerEdge* UPuzzleSequencerNode::GetEdge(UPuzzleSequencerNode* InChildNode)
{
	return Edges.Contains(InChildNode) ? Edges.FindChecked(InChildNode) : nullptr;
}

bool UPuzzleSequencerNode::IsLeafNode() const
{
	return ChildrenNodes.IsEmpty();
}

UPuzzleSequencer* UPuzzleSequencerNode::GetGraph() const
{
	return Graph;
}

FText UPuzzleSequencerNode::GetDescription_Implementation() const
{
	return LOCTEXT("NodeDesc", "Puzzle Sequencer Node");
}

bool UPuzzleSequencerNode::IsNameEditable() const
{
	return true;
}

FLinearColor UPuzzleSequencerNode::GetBackgroundColor() const
{
	return BackgroundColor;
}

FText UPuzzleSequencerNode::GetNodeTitle() const
{
	return NodeTitle.IsEmpty() ? GetDescription() : NodeTitle;
}

void UPuzzleSequencerNode::SetNodeTitle(const FText& NewTitle)
{
	NodeTitle = NewTitle;
}

bool UPuzzleSequencerNode::CanCreateConnection(UPuzzleSequencerNode* Other, FText& ErrorMessage)
{
	return true;
}

bool UPuzzleSequencerNode::CanCreateConnectionTo(UPuzzleSequencerNode* Other, int32 NumberOfChildrenNodes, FText& ErrorMessage)
{
	if (ParentLimitType == ENodeLimit::Limited && NumberOfChildrenNodes >= ParentLimit)
	{
		ErrorMessage = FText::FromString("Children limit exceeded");
		return false;
	}

	return true;
}

bool UPuzzleSequencerNode::CanCreateConnectionFrom(UPuzzleSequencerNode* Other, int32 NumberOfParentNodes, FText& ErrorMessage)
{
	if (ParentLimitType == ENodeLimit::Limited && NumberOfParentNodes >= ParentLimit)
	{
		ErrorMessage = FText::FromString("Parent limit exceeded");
		return false;
	}

	return true;
}

#undef LOCTEXT_NAMESPACE
