#include "AssetEditor/SEdNode_PSEEdge.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Images/SImage.h"
#include "Widgets/Text/SInlineEditableTextBlock.h"
#include "Widgets/SToolTip.h"
#include "SGraphPanel.h"
#include "EdGraphSchema_K2.h"
#include "PuzzleSequencerEdge.h"
#include "AssetEditor/EdNode_PSENode.h"
#include "AssetEditor/EdNode_PSEEdge.h"
#include "AssetEditor/ConnectionDrawingPolicy_PSE.h"

#define LOCTEXT_NAMESPACE "SPuzzleSequencerEdge"

void SEdNode_PSEEdge::Construct(const FArguments& InArgs, UEdNode_PSEEdge* InNode)
{
	this->GraphNode = InNode;
	this->UpdateGraphNode();
}

bool SEdNode_PSEEdge::RequiresSecondPassLayout() const
{
	return true;
}

void SEdNode_PSEEdge::PerformSecondPassLayout(const TMap<UObject*, TSharedRef<SNode>>& InNodeToWidgetLookup) const
{
	UEdNode_PSEEdge* EdgeNode = CastChecked<UEdNode_PSEEdge>(GraphNode);

	FGeometry StartGeom;
	FGeometry EndGeom;

	UEdNode_PSENode* Start = EdgeNode->GetStartNode();
	UEdNode_PSENode* End = EdgeNode->GetEndNode();
	if (Start != nullptr && End != nullptr)
	{
		const TSharedRef<SNode>* pFromWidget = InNodeToWidgetLookup.Find(Start);
		const TSharedRef<SNode>* pToWidget = InNodeToWidgetLookup.Find(End);
		if (pFromWidget != nullptr && pToWidget != nullptr)
		{
			const TSharedRef<SNode>& FromWidget = *pFromWidget;
			const TSharedRef<SNode>& ToWidget = *pToWidget;

			StartGeom = FGeometry(FVector2D(Start->NodePosX, Start->NodePosY), FVector2D::ZeroVector, FromWidget->GetDesiredSize(), 1.0f);
			EndGeom = FGeometry(FVector2D(End->NodePosX, End->NodePosY), FVector2D::ZeroVector, ToWidget->GetDesiredSize(), 1.0f);
		}
	}

	PositionBetweenTwoNodesWithOffset(StartGeom, EndGeom, 0, 1);
}

void SEdNode_PSEEdge::UpdateGraphNode()
{
	InputPins.Empty();
	OutputPins.Empty();

	RightNodeBox.Reset();
	LeftNodeBox.Reset();

	TSharedPtr<SNodeTitle> NodeTitle = SNew(SNodeTitle, GraphNode);

	this->ContentScale.Bind(this, &SGraphNode::GetContentScale);
	this->GetOrAddSlot(ENodeZone::Center)
	    .HAlign(HAlign_Center)
	    .VAlign(VAlign_Center)
	[
		SNew(SOverlay)
		+ SOverlay::Slot()
		[
			SNew(SImage)
				.Image(FEditorStyle::GetBrush("Graph.TransitionNode.ColorSpill"))
				.ColorAndOpacity(this, &SEdNode_PSEEdge::GetEdgeColour)
		]
		+ SOverlay::Slot()
		[
			SNew(SImage)
				.Image(this, &SEdNode_PSEEdge::GetEdgeImage)
				.Visibility(this, &SEdNode_PSEEdge::GetEdgeImageVisibility)
		]

		+ SOverlay::Slot()
		.Padding(FMargin(4.0f, 4.0f, 4.0f, 4.0f))
		[
			SNew(SVerticalBox)
			+ SVerticalBox::Slot()
			  .HAlign(HAlign_Center)
			  .AutoHeight()
			[
				SAssignNew(InlineEditableText, SInlineEditableTextBlock)
					.ColorAndOpacity(FLinearColor::Black)
					.Visibility(this, &SEdNode_PSEEdge::GetEdgeTitleVisibility)
					.Font(FCoreStyle::GetDefaultFontStyle("Regular", 12))
					.Text(NodeTitle.Get(), &SNodeTitle::GetHeadTitle)
					.OnTextCommitted(this, &SEdNode_PSEEdge::OnNameTextCommited)
			]
			+ SVerticalBox::Slot()
			.AutoHeight()
			[
				NodeTitle.ToSharedRef()
			]

		]
	];
}

void SEdNode_PSEEdge::PositionBetweenTwoNodesWithOffset(const FGeometry& InStartGeometry, const FGeometry& InEndGeometry, int32 InNodeIndex, int32 InMaxNodes) const
{
	// Get a reasonable seed point (halfway between the boxes)
	const FVector2D StartCenter = FGeometryHelper::CenterOf(InStartGeometry);
	const FVector2D EndCenter = FGeometryHelper::CenterOf(InEndGeometry);
	const FVector2D SeedPoint = (StartCenter + EndCenter) * 0.5f;

	// Find the (approximate) closest points between the two boxes
	const FVector2D StartAnchorPoint = FGeometryHelper::FindClosestPointOnGeom(InStartGeometry, SeedPoint);
	const FVector2D EndAnchorPoint = FGeometryHelper::FindClosestPointOnGeom(InEndGeometry, SeedPoint);

	// Position ourselves halfway along the connecting line between the nodes, elevated away perpendicular to the direction of the line
	const float Height = 30.0f;

	const FVector2D DesiredNodeSize = GetDesiredSize();

	FVector2D DeltaPos(EndAnchorPoint - StartAnchorPoint);

	if (DeltaPos.IsNearlyZero())
	{
		DeltaPos = FVector2D(10.0f, 0.0f);
	}

	const FVector2D Normal = FVector2D(DeltaPos.Y, -DeltaPos.X).GetSafeNormal();

	const FVector2D NewCenter = StartAnchorPoint + (0.5f * DeltaPos) + (Height * Normal);

	FVector2D DeltaNormal = DeltaPos.GetSafeNormal();

	// Calculate node offset in the case of multiple transitions between the same two nodes
	// MultiNodeOffset: the offset where 0 is the centre of the transition, -1 is 1 <size of node>
	// towards the PrevStateNode and +1 is 1 <size of node> towards the NextStateNode.

	const float MutliNodeSpace = 0.2f; // Space between multiple transition nodes (in units of <size of node> )
	const float MultiNodeStep = (1.f + MutliNodeSpace); //Step between node centres (Size of node + size of node spacer)

	const float MultiNodeStart = -((InMaxNodes - 1) * MultiNodeStep) / 2.f;
	const float MultiNodeOffset = MultiNodeStart + (InNodeIndex * MultiNodeStep);

	// Now we need to adjust the new center by the node size, zoom factor and multi node offset
	const FVector2D NewCorner = NewCenter - (0.5f * DesiredNodeSize) + (DeltaNormal * MultiNodeOffset * DesiredNodeSize.Size());

	GraphNode->NodePosX = NewCorner.X;
	GraphNode->NodePosY = NewCorner.Y;
}

void SEdNode_PSEEdge::OnNameTextCommited(const FText& InText, ETextCommit::Type InCommitInfo)
{
	SGraphNode::OnNameTextCommited(InText, InCommitInfo);

	UEdNode_PSEEdge* MyNode = CastChecked<UEdNode_PSEEdge>(GraphNode);

	if (MyNode != nullptr && MyNode->Edge != nullptr)
	{
		const FScopedTransaction Transaction(LOCTEXT("PuzzleSequencerEditorRenameEdge", "Puzzle Sequencer Editor: Rename Edge"));
		MyNode->Modify();
		MyNode->Edge->SetNodeTitle(InText);
		UpdateGraphNode();
	}
}

FSlateColor SEdNode_PSEEdge::GetEdgeColour() const
{
	UEdNode_PSEEdge* EdgeNode = CastChecked<UEdNode_PSEEdge>(GraphNode);
	if (EdgeNode != nullptr && EdgeNode->Edge != nullptr)
	{
		return EdgeNode->Edge->GetEdgeColour();
	}
	return FLinearColor(0.9f, 0.9f, 0.9f, 1.0f);
}

const FSlateBrush* SEdNode_PSEEdge::GetEdgeImage() const
{
	return FEditorStyle::GetBrush("Graph.TransitionNode.Icon");
}

EVisibility SEdNode_PSEEdge::GetEdgeImageVisibility() const
{
	UEdNode_PSEEdge* EdgeNode = CastChecked<UEdNode_PSEEdge>(GraphNode);
	if (EdgeNode && EdgeNode->Edge && EdgeNode->Edge->bShouldDrawTitle)
	{
		return EVisibility::Hidden;
	}

	return EVisibility::Visible;
}

EVisibility SEdNode_PSEEdge::GetEdgeTitleVisibility() const
{
	UEdNode_PSEEdge* EdgeNode = CastChecked<UEdNode_PSEEdge>(GraphNode);
	if (EdgeNode && EdgeNode->Edge && EdgeNode->Edge->bShouldDrawTitle)
	{
		return EVisibility::Visible;
	}

	return EVisibility::Collapsed;
}

#undef LOCTEXT_NAMESPACE
