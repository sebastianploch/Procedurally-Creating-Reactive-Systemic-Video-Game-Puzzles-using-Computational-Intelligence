#include "AssetEditor/SEdNode_PSENode.h"

#include "AssetEditor/EdNode_PSENode.h"
#include "AssetEditor/Colours_PSE.h"
#include "AssetEditor/PSEDragConnection.h"

#include "SGraphPin.h"
#include "SlateOptMacros.h"
#include "GraphEditorSettings.h"
#include "PuzzleSequencer.h"
#include "SCommentBubble.h"
#include "Widgets/Text/SInlineEditableTextBlock.h"

#define LOCTEXT_NAMESPACE "EdNode_PuzzleSequencer"

#pragma region Pin
class SPSEPin : public SGraphPin
{
	SLATE_BEGIN_ARGS(SPSEPin)
		{
		}

	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs, UEdGraphPin* InPin)
	{
		this->SetCursor(EMouseCursor::Default);
		bShowLabel = true;
		GraphPinObj = InPin;
		check(GraphPinObj != nullptr);

		const UEdGraphSchema* schema = GraphPinObj->GetSchema();
		check(schema);

		SBorder::Construct(SBorder::FArguments()
		                   .BorderImage(this, &SPSEPin::GetPinBorder)
		                   .BorderBackgroundColor(this, &SPSEPin::GetPinColor)
		                   .OnMouseButtonDown(this, &SPSEPin::OnPinMouseDown)
		                   .Padding(FMargin(5.f))
		);
	}

protected:
	virtual FSlateColor GetPinColor() const override
	{
		return PuzzleSequencerColours::Pin::Default;
	}

	virtual TSharedRef<SWidget> GetDefaultValueWidget() override
	{
		return SNew(STextBlock);
	}

	const FSlateBrush* GetPinBorder() const
	{
		return FEditorStyle::GetBrush(TEXT("PuzzleSequencer.StateNode.Body"));
	}

	virtual TSharedRef<FDragDropOperation> SpawnPinDragEvent(const TSharedRef<SGraphPanel>& InGraphPanel, const TArray<TSharedRef<SGraphPin>>& InStartingPins) override
	{
		FPSEDragConnection::FDraggedPinTable pinHandles;
		pinHandles.Reserve(InStartingPins.Num());
		for (const TSharedRef<SGraphPin>& pinWidget : InStartingPins)
		{
			pinHandles.Add(pinWidget->GetPinObj());
		}

		return FPSEDragConnection::New(InGraphPanel, pinHandles);
	}
};
#pragma endregion Pin

void SEdNode_PSENode::Construct(const FArguments& InArgs, UEdNode_PSENode* InNode)
{
	this->GraphNode = InNode;
	this->UpdateGraphNode();
	InNode->SEdNode = this;
}

BEGIN_SLATE_FUNCTION_BUILD_OPTIMIZATION

void SEdNode_PSENode::UpdateGraphNode()
{
	const FMargin NodePadding = FMargin(5);
	const FMargin NamePadding = FMargin(2);

	InputPins.Empty();
	OutputPins.Empty();

	// Reset variables that are going to be exposed, in case we are refreshing an already setup node.
	RightNodeBox.Reset();
	LeftNodeBox.Reset();

	const FSlateBrush* NodeTypeIcon = GetNameIcon();

	FLinearColor TitleShadowColor(0.6f, 0.6f, 0.6f);
	TSharedPtr<SErrorText> ErrorText;
	TSharedPtr<SVerticalBox> NodeBody;
	TSharedPtr<SNodeTitle> NodeTitle = SNew(SNodeTitle, GraphNode);

	this->ContentScale.Bind(this, &SGraphNode::GetContentScale);
	this->GetOrAddSlot(ENodeZone::Center)
	    .HAlign(HAlign_Fill)
	    .VAlign(VAlign_Center)
	[
		SNew(SBorder)
			.BorderImage(FEditorStyle::GetBrush("Graph.StateNode.Body"))
			.Padding(0.0f)
			.BorderBackgroundColor(this, &SEdNode_PSENode::GetBorderBackgroundColour)
		[
			SNew(SOverlay)

			+ SOverlay::Slot()
			  .HAlign(HAlign_Fill)
			  .VAlign(VAlign_Fill)
			[
				SNew(SVerticalBox)

				// Input Pin Area
				+ SVerticalBox::Slot()
				.FillHeight(1)
				[
					SAssignNew(LeftNodeBox, SVerticalBox)
				]

				// Output Pin Area	
				+ SVerticalBox::Slot()
				.FillHeight(1)
				[
					SAssignNew(RightNodeBox, SVerticalBox)
				]
			]

			+ SOverlay::Slot()
			  .HAlign(HAlign_Center)
			  .VAlign(VAlign_Center)
			  .Padding(8.0f)
			[
				SNew(SBorder)
					.BorderImage(FEditorStyle::GetBrush("Graph.StateNode.ColorSpill"))
					.BorderBackgroundColor(TitleShadowColor)
					.HAlign(HAlign_Center)
					.VAlign(VAlign_Center)
					.Visibility(EVisibility::SelfHitTestInvisible)
					.Padding(6.0f)
				[
					SAssignNew(NodeBody, SVerticalBox)

					// Title
					+ SVerticalBox::Slot()
					.AutoHeight()
					[
						SNew(SHorizontalBox)

						// Error message
						+ SHorizontalBox::Slot()
						.AutoWidth()
						[
							SAssignNew(ErrorText, SErrorText)
								.BackgroundColor(this, &SEdNode_PSENode::GetErrorColor)
								.ToolTipText(this, &SEdNode_PSENode::GetErrorMsgToolTip)
						]

						// Icon
						+ SHorizontalBox::Slot()
						  .AutoWidth()
						  .VAlign(VAlign_Center)
						[
							SNew(SImage)
							.Image(NodeTypeIcon)
						]

						// Node Title
						+ SHorizontalBox::Slot()
						.Padding(FMargin(4.0f, 0.0f, 4.0f, 0.0f))
						[
							SNew(SVerticalBox)
							+ SVerticalBox::Slot()
							.AutoHeight()
							[
								SAssignNew(InlineEditableText, SInlineEditableTextBlock)
									.Style(FEditorStyle::Get(), "Graph.StateNode.NodeTitleInlineEditableText")
									.Text(NodeTitle.Get(), &SNodeTitle::GetHeadTitle)
									.OnVerifyTextChanged(this, &SEdNode_PSENode::OnVerifyNameTextChanged)
									.OnTextCommitted(this, &SEdNode_PSENode::OnNameTextCommited)
									.IsReadOnly(this, &SEdNode_PSENode::IsNameReadOnly)
									.IsSelected(this, &SEdNode_PSENode::IsSelectedExclusively)
							]
							+ SVerticalBox::Slot()
							.AutoHeight()
							[
								NodeTitle.ToSharedRef()
							]
						]
					]
				]
			]
		]
	];

	// Create comment bubble
	TSharedPtr<SCommentBubble> CommentBubble;
	const FSlateColor CommentColor = GetDefault<UGraphEditorSettings>()->DefaultCommentNodeTitleColor;

	SAssignNew(CommentBubble, SCommentBubble)
		.GraphNode(GraphNode)
		.Text(this, &SGraphNode::GetNodeComment)
		.OnTextCommitted(this, &SGraphNode::OnCommentTextCommitted)
		.ColorAndOpacity(CommentColor)
		.AllowPinning(true)
		.EnableTitleBarBubble(true)
		.EnableBubbleCtrls(true)
		.GraphLOD(this, &SGraphNode::GetCurrentLOD)
		.IsGraphNodeHovered(this, &SGraphNode::IsHovered);

	GetOrAddSlot(ENodeZone::TopCenter)
		.SlotOffset(TAttribute<FVector2D>(CommentBubble.Get(), &SCommentBubble::GetOffset))
		.SlotSize(TAttribute<FVector2D>(CommentBubble.Get(), &SCommentBubble::GetSize))
		.AllowScaling(TAttribute<bool>(CommentBubble.Get(), &SCommentBubble::IsScalingAllowed))
		.VAlign(VAlign_Top)
		[
			CommentBubble.ToSharedRef()
		];

	ErrorReporting = ErrorText;
	ErrorReporting->SetError(ErrorMsg);
	CreatePinWidgets();
}

void SEdNode_PSENode::CreatePinWidgets()
{
	UEdNode_PSENode* StateNode = CastChecked<UEdNode_PSENode>(GraphNode);

	for (int32 PinIdx = 0; PinIdx < StateNode->Pins.Num(); PinIdx++)
	{
		UEdGraphPin* MyPin = StateNode->Pins[PinIdx];
		if (!MyPin->bHidden)
		{
			TSharedPtr<SGraphPin> NewPin = SNew(SPSEPin, MyPin);

			AddPin(NewPin.ToSharedRef());
		}
	}
}

void SEdNode_PSENode::AddPin(const TSharedRef<SGraphPin>& PinToAdd)
{
	PinToAdd->SetOwner(SharedThis(this));

	const UEdGraphPin* PinObj = PinToAdd->GetPinObj();
	const bool bAdvancedParameter = PinObj && PinObj->bAdvancedView;
	if (bAdvancedParameter)
	{
		PinToAdd->SetVisibility(TAttribute<EVisibility>(PinToAdd, &SGraphPin::IsPinVisibleAsAdvanced));
	}

	TSharedPtr<SVerticalBox> PinBox;
	if (PinToAdd->GetDirection() == EEdGraphPinDirection::EGPD_Input)
	{
		PinBox = LeftNodeBox;
		InputPins.Add(PinToAdd);
	}
	else // Direction == EEdGraphPinDirection::EGPD_Output
	{
		PinBox = RightNodeBox;
		OutputPins.Add(PinToAdd);
	}

	if (PinBox)
	{
		PinBox->AddSlot()
		      .HAlign(HAlign_Fill)
		      .VAlign(VAlign_Fill)
		      .FillHeight(1.0f)
			//.Padding(6.0f, 0.0f)
			[
				PinToAdd
			];
	}
}

bool SEdNode_PSENode::IsNameReadOnly() const
{
	UEdNode_PSENode* EdNode_Node = Cast<UEdNode_PSENode>(GraphNode);
	check(EdNode_Node != nullptr);

	UPuzzleSequencer* GenericGraph = EdNode_Node->Node->Graph;
	check(GenericGraph != nullptr);

	return (!GenericGraph->bCanRenameNode || !EdNode_Node->Node->IsNameEditable()) || SGraphNode::IsNameReadOnly();
}

END_SLATE_FUNCTION_BUILD_OPTIMIZATION

void SEdNode_PSENode::OnNameTextCommited(const FText& InText, ETextCommit::Type InCommitInfo)
{
	SGraphNode::OnNameTextCommited(InText, InCommitInfo);

	UEdNode_PSENode* node = CastChecked<UEdNode_PSENode>(GraphNode);

	if (node && node->Node)
	{
		const FScopedTransaction transaction(LOCTEXT("PuzzleSequencerEditorRenameNode", "Puzzle Sequencer Editor: Rename Node"));
		node->Modify();
		node->Node->Modify();
		node->Node->SetNodeTitle(InText);
		UpdateGraphNode();
	}
}

FSlateColor SEdNode_PSENode::GetBorderBackgroundColour() const
{
	UEdNode_PSENode* node = CastChecked<UEdNode_PSENode>(GraphNode);
	return node ? node->GetBackgroundColour() : PuzzleSequencerColours::NodeBorder::HighlightAbortRange0;
}

FSlateColor SEdNode_PSENode::GetBackgroundColour() const
{
	return PuzzleSequencerColours::NodeBody::Default;
}

EVisibility SEdNode_PSENode::GetDragOverMarkerVisibility() const
{
	return EVisibility::Visible;
}

const FSlateBrush* SEdNode_PSENode::GetNameIcon() const
{
	return FEditorStyle::GetBrush(TEXT("BTEditor.Graph.BTNode.Icon"));
}

#undef LOCTEXT_NAMESPACE
