#include "AssetEditor/PSEDragConnection.h"
#include "AssetEditor/EdNode_PSENode.h"

#include "SGraphPanel.h"
#include "ScopedTransaction.h"
#include "EdGraph/EdGraph.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Images/SImage.h"
#include "Framework/Application/SlateApplication.h"

FPSEDragConnection::FPSEDragConnection(const TSharedRef<SGraphPanel>& InGraphPanel, const FDraggedPinTable& InDraggedPins)
	: GraphPanel(InGraphPanel),
	  DraggingPins(InDraggedPins),
	  DecoratorAdjust(FSlateApplication::Get().GetCursorSize())
{
	if (!DraggingPins.IsEmpty())
	{
		const UEdGraphPin* pinObj = FDraggedPinTable::TConstIterator(InDraggedPins)->GetPinObj(*InGraphPanel);
		if (pinObj && pinObj->Direction == EGPD_Input)
		{
			DecoratorAdjust *= FVector2D(-1.0, 1.0);
		}
	}

	for (const FGraphPinHandle& draggedPin : InDraggedPins)
	{
		InGraphPanel->OnBeginMakingConnection(draggedPin);
	}
}

TSharedRef<FPSEDragConnection> FPSEDragConnection::New(const TSharedRef<SGraphPanel>& InGraphPanel, const FDraggedPinTable& InDraggedPins)
{
	TSharedRef<FPSEDragConnection> operation = MakeShareable<FPSEDragConnection>(new FPSEDragConnection(InGraphPanel, InDraggedPins));
	operation->Construct();

	return operation;
}

void FPSEDragConnection::OnDrop(bool bDropWasHandled, const FPointerEvent& MouseEvent)
{
	GraphPanel->OnStopMakingConnection();
	FGraphEditorDragDropAction::OnDrop(bDropWasHandled, MouseEvent);
}

void FPSEDragConnection::OnDragged(const FDragDropEvent& DragDropEvent)
{
	const FVector2D targetPosition = DragDropEvent.GetScreenSpacePosition();
	CursorDecoratorWindow->MoveWindowTo(DragDropEvent.GetScreenSpacePosition() + DecoratorAdjust);
	GraphPanel->RequestDeferredPan(targetPosition);
}

void FPSEDragConnection::HoverTargetChanged()
{
	TArray<FPinConnectionResponse> uniqueMessages;

	if (UEdGraphPin* targetPinObj = GetHoveredPin())
	{
		TArray<UEdGraphPin*> validSourcePins;
		ValidateGraphPinList(validSourcePins);

		for (UEdGraphPin* pinObj : validSourcePins)
		{
			UEdGraph* graphObj = pinObj->GetOwningNode()->GetGraph();

			const FPinConnectionResponse response = graphObj->GetSchema()->CanCreateConnection(pinObj, targetPinObj);
			if (response.Response == ECanCreateConnectionResponse::CONNECT_RESPONSE_DISALLOW)
			{
				TSharedPtr<SGraphNode> nodeWidget = targetPinObj->GetOwningNode()->DEPRECATED_NodeWidget.Pin();
				if (nodeWidget.IsValid())
				{
					nodeWidget->NotifyDisallowedPinConnection(pinObj, targetPinObj);
				}
			}

			uniqueMessages.AddUnique(response);
		}
	}
	else if (UEdNode_PSENode* targetNodeObj = Cast<UEdNode_PSENode>(GetHoveredNode()))
	{
		TArray<UEdGraphPin*> validSourcePins;
		ValidateGraphPinList(validSourcePins);

		for (UEdGraphPin* pinObj : validSourcePins)
		{
			FPinConnectionResponse response;
			FText responseText;

			const UEdGraphSchema* schema = pinObj->GetSchema();
			UEdGraphPin* targetPin = targetNodeObj->GetInputPin();

			if (schema && targetPin)
			{
				response = schema->CanCreateConnection(pinObj, targetPin);
				if (response.Response == ECanCreateConnectionResponse::CONNECT_RESPONSE_DISALLOW)
				{
					TSharedPtr<SGraphNode> nodeWidget = targetPin->GetOwningNode()->DEPRECATED_NodeWidget.Pin();
					if (nodeWidget.IsValid())
					{
						nodeWidget->NotifyDisallowedPinConnection(pinObj, targetPinObj);
					}
				}
			}
			else
			{
				response = FPinConnectionResponse(CONNECT_RESPONSE_DISALLOW, NSLOCTEXT("AssetSchema_PuzzleSequencer", "PinError", "Not a valid UPSEEdNode"));
			}

			uniqueMessages.AddUnique(response);
		}
	}
	else if (UEdGraph* currentHoveredGraph = GetHoveredGraph())
	{
		TArray<UEdGraphPin*> validSourcePins;
		ValidateGraphPinList(validSourcePins);

		for (UEdGraphPin* pinObj : validSourcePins)
		{
			FPinConnectionResponse response = currentHoveredGraph->GetSchema()->CanCreateNewNodes(pinObj);
			if (!response.Message.IsEmpty())
			{
				uniqueMessages.AddUnique(response);
			}
		}
	}

	if (uniqueMessages.IsEmpty())
	{
		SetSimpleFeedbackMessage(
			FEditorStyle::GetBrush("Graph.ConnectorFeedback.NewNode"),
			FLinearColor::White,
			NSLOCTEXT("GraphEditor.Feedback", "PlaceNewNode", "Place a new node."));
	}
	else
	{
		TSharedRef<SVerticalBox> feedbackBox = SNew(SVerticalBox);
		for (auto response : uniqueMessages)
		{
			const FSlateBrush* statusSymbol = nullptr;

			switch (response.Response)
			{
			case CONNECT_RESPONSE_MAKE:
			case CONNECT_RESPONSE_BREAK_OTHERS_A:
			case CONNECT_RESPONSE_BREAK_OTHERS_B:
			case CONNECT_RESPONSE_BREAK_OTHERS_AB:
				statusSymbol = FEditorStyle::GetBrush(TEXT("Graph.ConnectorFeedback.OK"));
				break;

			case CONNECT_RESPONSE_MAKE_WITH_CONVERSION_NODE:
				statusSymbol = FEditorStyle::GetBrush(TEXT("Graph.ConnectorFeedback.ViaCast"));
				break;

			case CONNECT_RESPONSE_DISALLOW:
			default:
				statusSymbol = FEditorStyle::GetBrush(TEXT("Graph.ConnectorFeedback.Error"));
				break;
			}

			feedbackBox->AddSlot()
			           .AutoHeight()
			[
				SNew(SHorizontalBox)
				+ SHorizontalBox::Slot()
				  .AutoWidth()
				  .Padding(3.f)
				  .VAlign(VAlign_Center)
				[
					SNew(SImage).Image(statusSymbol)
				]
				+ SHorizontalBox::Slot()
				  .AutoWidth()
				  .VAlign(VAlign_Center)
				[
					SNew(STextBlock).Text(response.Message)
				]
			];
		}

		SetFeedbackMessage(feedbackBox);
	}
}

FReply FPSEDragConnection::DroppedOnPin(FVector2D ScreenPosition, FVector2D GraphPosition)
{
	TArray<UEdGraphPin*> validSourcePins;
	ValidateGraphPinList(validSourcePins);

	const FScopedTransaction transaction(NSLOCTEXT("UnrealEd", "GraphEd_CreateConnection", "Create Pin Link"));

	UEdGraphPin* pinB = GetHoveredPin();
	bool bError = false;
	TSet<UEdGraphNode*> nodeList;

	for (UEdGraphPin* pinA : validSourcePins)
	{
		if (pinA && pinB)
		{
			UEdGraph* graphObj = pinA->GetOwningNode()->GetGraph();

			if (graphObj->GetSchema()->TryCreateConnection(pinA, pinB))
			{
				if (!pinA->IsPendingKill())
				{
					nodeList.Add(pinA->GetOwningNode());
				}
				if (!pinB->IsPendingKill())
				{
					nodeList.Add(pinB->GetOwningNode());
				}
			}
		}
		else
		{
			bError = true;
		}
	}

	for (UEdGraphNode* node : nodeList)
	{
		node->NodeConnectionListChanged();
	}

	if (bError)
	{
		return FReply::Unhandled();
	}

	return FReply::Handled();
}

FReply FPSEDragConnection::DroppedOnNode(FVector2D ScreenPosition, FVector2D GraphPosition)
{
	bool bHandledPinDropOnNode = false;
	UEdGraphNode* nodeOver = GetHoveredNode();

	if (nodeOver)
	{
		TArray<UEdGraphPin*> validSourcePins;
		ValidateGraphPinList(validSourcePins);

		if (!validSourcePins.IsEmpty())
		{
			for (UEdGraphPin* sourcePin : validSourcePins)
			{
				FText responseText;
				if (sourcePin->GetOwningNode() != nodeOver
					&& sourcePin->GetSchema()->SupportsDropPinOnNode(nodeOver, sourcePin->PinType, sourcePin->Direction, responseText))
				{
					bHandledPinDropOnNode = true;

					const FName pinName = sourcePin->PinFriendlyName.IsEmpty() ? sourcePin->PinName : *sourcePin->PinFriendlyName.ToString();
					const FScopedTransaction transaction((sourcePin->Direction == EGPD_Output) ? NSLOCTEXT("UnrealEd", "AddInParam", "Add In Parameter") : NSLOCTEXT("UnrealEd", "AddOutParam", "Add Out Parameter"));

					UEdGraphPin* edGraphPin = nodeOver->GetSchema()->DropPinOnNode(GetHoveredNode(), pinName, sourcePin->PinType, sourcePin->Direction);

					if (sourcePin->GetOwningNodeUnchecked() && edGraphPin)
					{
						sourcePin->Modify();
						edGraphPin->Modify();
						sourcePin->GetSchema()->TryCreateConnection(sourcePin, edGraphPin);
					}
				}

				if (!bHandledPinDropOnNode && !responseText.IsEmpty())
				{
					bHandledPinDropOnNode = true;
				}
			}
		}
	}

	return bHandledPinDropOnNode ? FReply::Handled() : FReply::Unhandled();
}

FReply FPSEDragConnection::DroppedOnPanel(const TSharedRef<SWidget>& Panel, FVector2D ScreenPosition, FVector2D GraphPosition, UEdGraph& Graph)
{
	TArray<UEdGraphPin*> pinObjects;
	ValidateGraphPinList(pinObjects);

	TSharedPtr<SWidget> widgetToFocus = GraphPanel->SummonContextMenu(ScreenPosition, GraphPosition, nullptr, nullptr, pinObjects);

	return (widgetToFocus.IsValid())
		       ? FReply::Handled().SetUserFocus(widgetToFocus.ToSharedRef(), EFocusCause::SetDirectly)
		       : FReply::Handled();
}

void FPSEDragConnection::ValidateGraphPinList(TArray<UEdGraphPin*>& OutValidPins)
{
	OutValidPins.Empty(DraggingPins.Num());
	for (const FGraphPinHandle& pinHandle : DraggingPins)
	{
		if (UEdGraphPin* graphPin = pinHandle.GetPinObj(*GraphPanel))
		{
			OutValidPins.Add(graphPin);
		}
	}
}

