#include "PuzzleSequencerNode.h"
#include "PuzzleSequencer.h"
#include "PuzzleActor.h"
#include "PuzzleSequencerLog.h"

#define LOCTEXT_NAMESPACE "PuzzleSequencerNode"

UPuzzleSequencerNode::UPuzzleSequencerNode()
{
#if WITH_EDITORONLY_DATA
	NodeTitle = FText::FromString(TEXT("Puzzle Sequencer Node"));
	CompatibleGraphType = UPuzzleSequencer::StaticClass();
	BackgroundColor = FLinearColor::Black;
	ContextMenuName = FText::FromString(TEXT("Puzzle Sequencer Node"));
	ParentLimitType = EPSENodeLimit::Limited;
	ParentLimit = 1;
	ChildrenLimitType = EPSENodeLimit::Limited;
	ChildrenLimit = 1;
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

#if WITH_EDITOR
void UPuzzleSequencerNode::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	if (!PropertyChangedEvent.Property)
	{
		return;
	}

	// Verity that changed property is for puzzle actor
	const FName propertyName = PropertyChangedEvent.Property->GetFName();
	if (propertyName == GET_MEMBER_NAME_CHECKED(UPuzzleSequencerNode, PuzzleActor))
	{
		UpdatePuzzleActorInformation();
	}

	Super::PostEditChangeProperty(PropertyChangedEvent);
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
	if (ParentLimitType == EPSENodeLimit::Limited && NumberOfChildrenNodes >= ParentLimit)
	{
		ErrorMessage = FText::FromString("Children limit exceeded");
		return false;
	}

	return true;
}

bool UPuzzleSequencerNode::CanCreateConnectionFrom(UPuzzleSequencerNode* Other, int32 NumberOfParentNodes, FText& ErrorMessage)
{
	if (ParentLimitType == EPSENodeLimit::Limited && NumberOfParentNodes >= ParentLimit)
	{
		ErrorMessage = FText::FromString("Parent limit exceeded");
		return false;
	}

	return true;
}
#endif // WITH_EDITOR

void UPuzzleSequencerNode::UpdatePuzzleActorInformation()
{
	if (!PuzzleActor.IsValid())
	{
		ResetPuzzleActorInformation();
		return;
	}

	const auto* actor = PuzzleActor.LoadSynchronous();

#if WITH_EDITOR
	NodeTitle = FText::FromString(actor->GetName());

	// Background colour
	const EPuzzleType puzzleType = actor->GetPuzzleType();
	switch (puzzleType)
	{
	case EPuzzleType::Start:
		BackgroundColor = FColor::Green;
		break;
	case EPuzzleType::Intermediate:
		BackgroundColor = FColor::Orange;
		break;
	case EPuzzleType::Finish:
		BackgroundColor = FColor::Red;
		break;
	case EPuzzleType::None:
		BackgroundColor = FColor::Black;
		break;
	default:
		UE_LOG(LogPuzzleSequencer, Warning, TEXT("%s - using undefined enum {(Role %s)}"), *FString(__FUNCTION__), *UEnum::GetDisplayValueAsText(puzzleType).ToString())
		break;
	}
#endif // WITH_EDITOR
}

void UPuzzleSequencerNode::ResetPuzzleActorInformation()
{
#if WITH_EDITOR
	NodeTitle = FText::FromString(TEXT("Puzzle Sequencer Node"));
	BackgroundColor = FColor::Black;
#endif // WITH_EDITOR
}

#undef LOCTEXT_NAMESPACE
