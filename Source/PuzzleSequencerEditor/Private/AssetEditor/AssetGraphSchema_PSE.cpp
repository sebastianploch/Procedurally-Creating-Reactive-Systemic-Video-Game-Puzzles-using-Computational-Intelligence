#include "AssetEditor/AssetGraphSchema_PSE.h"
#include "ToolMenus.h"
#include "AssetEditor/EdNode_PSENode.h"
#include "AssetEditor/EdNode_PSEEdge.h"
#include "AssetEditor/ConnectionDrawingPolicy_PSE.h"
#include "GraphEditorActions.h"
#include "AssetEditor/EdGraph_PSE.h"
#include "Framework/Commands/GenericCommands.h"

#define LOCTEXT_NAMESPACE "AssetSchema_PuzzleSequencer"

#pragma region NodeVisitorCycleChecker
class FNodeVisitorCycleChecker
{
public:
	bool CheckForLoop(UEdGraphNode* InStartNode, UEdGraphNode* InEndNode)
	{
		VisitedNodes.Add(InStartNode);
		return TraverseNodes(InEndNode);
	}

private:
	bool TraverseNodes(UEdGraphNode* InNode)
	{
		VisitedNodes.Add(InNode);
		for (const auto& pin : InNode->Pins)
		{
			if (pin->Direction == EGPD_Output)
			{
				for (const auto& otherPin : pin->LinkedTo)
				{
					UEdGraphNode* otherNode = otherPin->GetOwningNode();
					if (VisitedNodes.Contains(otherNode))
					{
						return false;
					}
					else if (!FinishedNodes.Contains(otherNode))
					{
						if (!TraverseNodes(otherNode))
						{
							return false;
						}
					}
				}
			}
		}

		VisitedNodes.Remove(InNode);
		FinishedNodes.Add(InNode);
		return true;
	}

private:
	TSet<UEdGraphNode*> VisitedNodes{};
	TSet<UEdGraphNode*> FinishedNodes{};
};
#pragma endregion NodeVisitorCycleChecker

#pragma region NewNode
FAssetSchemaAction_PSE_NewNode::FAssetSchemaAction_PSE_NewNode(const FText& InNodeCategory, const FText& InMenuDesc, const FText& InToolTip, const int32 InGrouping)
	: FEdGraphSchemaAction(InNodeCategory, InMenuDesc, InToolTip, InGrouping)
{
}

UEdGraphNode* FAssetSchemaAction_PSE_NewNode::PerformAction(UEdGraph* ParentGraph, UEdGraphPin* FromPin, const FVector2D Location, bool bSelectNewNode)
{
	UEdGraphNode* resultNode = nullptr;

	if (NodeTemplate)
	{
		const FScopedTransaction transaction(LOCTEXT("PuzzleSequencerEditorNewNode", "Puzzle Sequencer Editor: New Node"));
		ParentGraph->Modify();
		if (FromPin)
		{
			FromPin->Modify();
		}

		NodeTemplate->Rename(nullptr, ParentGraph);
		ParentGraph->AddNode(NodeTemplate, true, bSelectNewNode);

		NodeTemplate->CreateNewGuid();
		NodeTemplate->PostPlacedNewNode();
		NodeTemplate->AllocateDefaultPins();
		NodeTemplate->AutowireNewNode(FromPin);

		NodeTemplate->NodePosX = Location.X;
		NodeTemplate->NodePosY = Location.Y;

		NodeTemplate->Node->SetFlags(RF_Transactional);
		NodeTemplate->SetFlags(RF_Transactional);

		resultNode = NodeTemplate;
	}

	return resultNode;
}

void FAssetSchemaAction_PSE_NewNode::AddReferencedObjects(FReferenceCollector& Collector)
{
	FEdGraphSchemaAction::AddReferencedObjects(Collector);
	Collector.AddReferencedObject(NodeTemplate);
}
#pragma endregion NewNode

#pragma region NewEdge
FAssetSchemaAction_PSE_NewEdge::FAssetSchemaAction_PSE_NewEdge(const FText& InNodeCategory, const FText& InMenuDesc, const FText& InToolTip, const int32 InGrouping)
	: FEdGraphSchemaAction(InNodeCategory, InMenuDesc, InToolTip, InGrouping)
{
}

UEdGraphNode* FAssetSchemaAction_PSE_NewEdge::PerformAction(UEdGraph* ParentGraph, UEdGraphPin* FromPin, const FVector2D Location, bool bSelectNewNode)
{
	UEdGraphNode* resultNode = nullptr;

	if (NodeTemplate)
	{
		const FScopedTransaction transaction(LOCTEXT("PuzzleSequencerEditorNewEdge", "Puzzle Sequencer Editor: New Edge"));
		ParentGraph->Modify();
		if (FromPin)
		{
			FromPin->Modify();
		}

		NodeTemplate->Rename(nullptr, ParentGraph);
		ParentGraph->AddNode(NodeTemplate, true, bSelectNewNode);

		NodeTemplate->CreateNewGuid();
		NodeTemplate->PostPlacedNewNode();
		NodeTemplate->AllocateDefaultPins();
		NodeTemplate->AutowireNewNode(FromPin);

		NodeTemplate->NodePosX = Location.X;
		NodeTemplate->NodePosY = Location.Y;

		NodeTemplate->Edge->SetFlags(RF_Transactional);
		NodeTemplate->SetFlags(RF_Transactional);

		resultNode = NodeTemplate;
	}

	return resultNode;
}

void FAssetSchemaAction_PSE_NewEdge::AddReferencedObjects(FReferenceCollector& Collector)
{
	FEdGraphSchemaAction::AddReferencedObjects(Collector);
	Collector.AddReferencedObject(NodeTemplate);
}
#pragma endregion NewEdge

void UAssetGraphSchema_PSE::GetBreakLinkToSubMenuActions(UToolMenu* Menu, UEdGraphPin* InGraphPin)
{
	// Make sure we have a unique name for every entry in the list
	TMap<FString, uint32> LinkTitleCount;

	FToolMenuSection& Section = Menu->FindOrAddSection("PuzzleSequencerAssetGraphSchemaPinActions");

	// Add all the links we could break from
	for (TArray<class UEdGraphPin*>::TConstIterator Links(InGraphPin->LinkedTo); Links; ++Links)
	{
		UEdGraphPin* Pin = *Links;
		FString TitleString = Pin->GetOwningNode()->GetNodeTitle(ENodeTitleType::ListView).ToString();
		FText Title = FText::FromString(TitleString);
		if (Pin->PinName != TEXT(""))
		{
			TitleString = FString::Printf(TEXT("%s (%s)"), *TitleString, *Pin->PinName.ToString());

			// Add name of connection if possible
			FFormatNamedArguments Args;
			Args.Add(TEXT("NodeTitle"), Title);
			Args.Add(TEXT("PinName"), Pin->GetDisplayName());
			Title = FText::Format(LOCTEXT("BreakDescPin", "{NodeTitle} ({PinName})"), Args);
		}

		uint32& Count = LinkTitleCount.FindOrAdd(TitleString);

		FText Description;
		FFormatNamedArguments Args;
		Args.Add(TEXT("NodeTitle"), Title);
		Args.Add(TEXT("NumberOfNodes"), Count);

		if (Count == 0)
		{
			Description = FText::Format(LOCTEXT("BreakDesc", "Break link to {NodeTitle}"), Args);
		}
		else
		{
			Description = FText::Format(LOCTEXT("BreakDescMulti", "Break link to {NodeTitle} ({NumberOfNodes})"), Args);
		}
		++Count;

		Section.AddMenuEntry(NAME_None, Description, Description, FSlateIcon(), FUIAction(
			                     FExecuteAction::CreateUObject(this, &UAssetGraphSchema_PSE::BreakSinglePinLink, const_cast<UEdGraphPin*>(InGraphPin), *Links)));
	}
}

EGraphType UAssetGraphSchema_PSE::GetGraphType(const UEdGraph* TestEdGraph) const
{
	return GT_StateMachine;
}

void UAssetGraphSchema_PSE::GetGraphContextActions(FGraphContextMenuBuilder& ContextMenuBuilder) const
{
	UPuzzleSequencer* Graph = CastChecked<UPuzzleSequencer>(ContextMenuBuilder.CurrentGraph->GetOuter());

	if (Graph->NodeType == nullptr)
	{
		return;
	}

	const bool bNoParent = (ContextMenuBuilder.FromPin == NULL);

	const FText AddToolTip = LOCTEXT("NewPuzzleSequencerNodeTooltip", "Add node here");

	TSet<TSubclassOf<UPuzzleSequencerNode>> Visited;

	FText Desc = Graph->NodeType.GetDefaultObject()->ContextMenuName;

	if (Desc.IsEmpty())
	{
		FString Title = Graph->NodeType->GetName();
		Title.RemoveFromEnd("_C");
		Desc = FText::FromString(Title);
	}

	if (!Graph->NodeType->HasAnyClassFlags(CLASS_Abstract))
	{
		TSharedPtr<FAssetSchemaAction_PSE_NewNode> NewNodeAction(new FAssetSchemaAction_PSE_NewNode(LOCTEXT("PuzzleSequencerNodeAction", "Puzzle Sequencer Node"), Desc, AddToolTip, 0));
		NewNodeAction->NodeTemplate = NewObject<UEdNode_PSENode>(ContextMenuBuilder.OwnerOfTemporaries);
		NewNodeAction->NodeTemplate->Node = NewObject<UPuzzleSequencerNode>(NewNodeAction->NodeTemplate, Graph->NodeType);
		NewNodeAction->NodeTemplate->Node->Graph = Graph;
		ContextMenuBuilder.AddAction(NewNodeAction);

		Visited.Add(Graph->NodeType);
	}

	for (TObjectIterator<UClass> It; It; ++It)
	{
		if (It->IsChildOf(Graph->NodeType) && !It->HasAnyClassFlags(CLASS_Abstract) && !Visited.Contains(*It))
		{
			TSubclassOf<UPuzzleSequencerNode> NodeType = *It;

			if (It->GetName().StartsWith("REINST") || It->GetName().StartsWith("SKEL"))
			{
				continue;
			}

			if (!Graph->GetClass()->IsChildOf(NodeType.GetDefaultObject()->CompatibleGraphType))
			{
				continue;
			}

			Desc = NodeType.GetDefaultObject()->ContextMenuName;

			if (Desc.IsEmpty())
			{
				FString Title = NodeType->GetName();
				Title.RemoveFromEnd("_C");
				Desc = FText::FromString(Title);
			}

			TSharedPtr<FAssetSchemaAction_PSE_NewNode> Action(new FAssetSchemaAction_PSE_NewNode(LOCTEXT("PuzzleSequencerNodeAction", "Puzzle Sequencer Node"), Desc, AddToolTip, 0));
			Action->NodeTemplate = NewObject<UEdNode_PSENode>(ContextMenuBuilder.OwnerOfTemporaries);
			Action->NodeTemplate->Node = NewObject<UPuzzleSequencerNode>(Action->NodeTemplate, NodeType);
			Action->NodeTemplate->Node->Graph = Graph;
			ContextMenuBuilder.AddAction(Action);

			Visited.Add(NodeType);
		}
	}
}

void UAssetGraphSchema_PSE::GetContextMenuActions(UToolMenu* Menu, UGraphNodeContextMenuContext* Context) const
{
	if (Context->Pin)
	{
		{
			FToolMenuSection& Section = Menu->AddSection("PuzzleSequencerAssetGraphSchemaNodeActions", LOCTEXT("PinActionsMenuHeader", "Pin Actions"));
			// Only display the 'Break Links' option if there is a link to break!
			if (Context->Pin->LinkedTo.Num() > 0)
			{
				Section.AddMenuEntry(FGraphEditorCommands::Get().BreakPinLinks);

				// add sub menu for break link to
				if (Context->Pin->LinkedTo.Num() > 1)
				{
					Section.AddSubMenu(
						"BreakLinkTo",
						LOCTEXT("BreakLinkTo", "Break Link To..."),
						LOCTEXT("BreakSpecificLinks", "Break a specific link..."),
						FNewToolMenuDelegate::CreateUObject((UAssetGraphSchema_PSE* const)this, &UAssetGraphSchema_PSE::GetBreakLinkToSubMenuActions, const_cast<UEdGraphPin*>(Context->Pin)));
				}
				else
				{
					((UAssetGraphSchema_PSE* const)this)->GetBreakLinkToSubMenuActions(Menu, const_cast<UEdGraphPin*>(Context->Pin));
				}
			}
		}
	}
	else if (Context->Node)
	{
		{
			FToolMenuSection& Section = Menu->AddSection("PuzzleSequencerAssetGraphSchemaNodeActions", LOCTEXT("ClassActionsMenuHeader", "Node Actions"));
			Section.AddMenuEntry(FGenericCommands::Get().Delete);
			Section.AddMenuEntry(FGenericCommands::Get().Cut);
			Section.AddMenuEntry(FGenericCommands::Get().Copy);
			Section.AddMenuEntry(FGenericCommands::Get().Duplicate);

			Section.AddMenuEntry(FGraphEditorCommands::Get().BreakNodeLinks);
		}
	}

	Super::GetContextMenuActions(Menu, Context);
}

const FPinConnectionResponse UAssetGraphSchema_PSE::CanCreateConnection(const UEdGraphPin* A, const UEdGraphPin* B) const
{
	// Make sure the pins are not on the same node
	if (A->GetOwningNode() == B->GetOwningNode())
	{
		return FPinConnectionResponse(CONNECT_RESPONSE_DISALLOW, LOCTEXT("PinErrorSameNode", "Can't connect node to itself"));
	}

	const UEdGraphPin* Out = A;
	const UEdGraphPin* In = B;

	UEdNode_PSENode* EdNode_Out = Cast<UEdNode_PSENode>(Out->GetOwningNode());
	UEdNode_PSENode* EdNode_In = Cast<UEdNode_PSENode>(In->GetOwningNode());

	if (EdNode_Out == nullptr || EdNode_In == nullptr)
	{
		return FPinConnectionResponse(CONNECT_RESPONSE_DISALLOW, LOCTEXT("PinError", "Not a valid UPSEEdNode"));
	}

	//Determine if we can have cycles or not
	bool bAllowCycles = false;
	auto EdGraph = Cast<UEdGraph_PSE>(Out->GetOwningNode()->GetGraph());
	if (EdGraph != nullptr)
	{
		bAllowCycles = EdGraph->GetGraph()->bCanBeCyclical;
	}

	// check for cycles
	FNodeVisitorCycleChecker CycleChecker;
	if (!bAllowCycles && !CycleChecker.CheckForLoop(Out->GetOwningNode(), In->GetOwningNode()))
	{
		return FPinConnectionResponse(CONNECT_RESPONSE_DISALLOW, LOCTEXT("PinErrorCycle", "Can't create a graph cycle"));
	}

	FText ErrorMessage;
	if (!EdNode_Out->Node->CanCreateConnectionTo(EdNode_In->Node, EdNode_Out->GetOutputPin()->LinkedTo.Num(), ErrorMessage))
	{
		return FPinConnectionResponse(CONNECT_RESPONSE_DISALLOW, ErrorMessage);
	}
	if (!EdNode_In->Node->CanCreateConnectionFrom(EdNode_Out->Node, EdNode_In->GetInputPin()->LinkedTo.Num(), ErrorMessage))
	{
		return FPinConnectionResponse(CONNECT_RESPONSE_DISALLOW, ErrorMessage);
	}


	if (EdNode_Out->Node->GetGraph()->bEdgeEnabled)
	{
		return FPinConnectionResponse(CONNECT_RESPONSE_MAKE_WITH_CONVERSION_NODE, LOCTEXT("PinConnect", "Connect nodes with edge"));
	}
	else
	{
		return FPinConnectionResponse(CONNECT_RESPONSE_MAKE, LOCTEXT("PinConnect", "Connect nodes"));
	}
}

bool UAssetGraphSchema_PSE::TryCreateConnection(UEdGraphPin* A, UEdGraphPin* B) const
{
	// We don't actually care about the pin, we want the node that is being dragged between
	UEdNode_PSENode* NodeA = Cast<UEdNode_PSENode>(A->GetOwningNode());
	UEdNode_PSENode* NodeB = Cast<UEdNode_PSENode>(B->GetOwningNode());

	// Check that this edge doesn't already exist
	for (UEdGraphPin* TestPin : NodeA->GetOutputPin()->LinkedTo)
	{
		UEdGraphNode* ChildNode = TestPin->GetOwningNode();
		if (UEdNode_PSEEdge* EdNode_Edge = Cast<UEdNode_PSEEdge>(ChildNode))
		{
			ChildNode = EdNode_Edge->GetEndNode();
		}

		if (ChildNode == NodeB)
		{
			return false;
		}
	}

	if (NodeA && NodeB)
	{
		// Always create connections from node A to B, don't allow adding in reverse
		Super::TryCreateConnection(NodeA->GetOutputPin(), NodeB->GetInputPin());
		return true;
	}
	else
	{
		return false;
	}
}

bool UAssetGraphSchema_PSE::CreateAutomaticConversionNodeAndConnections(UEdGraphPin* A, UEdGraphPin* B) const
{
	UEdNode_PSENode* NodeA = Cast<UEdNode_PSENode>(A->GetOwningNode());
	UEdNode_PSENode* NodeB = Cast<UEdNode_PSENode>(B->GetOwningNode());

	// Are nodes and pins all valid?
	if (!NodeA || !NodeA->GetOutputPin() || !NodeB || !NodeB->GetInputPin())
	{
		return false;
	}

	UPuzzleSequencer* Graph = NodeA->Node->GetGraph();

	FVector2D InitPos((NodeA->NodePosX + NodeB->NodePosX) / 2, (NodeA->NodePosY + NodeB->NodePosY) / 2);

	FAssetSchemaAction_PSE_NewEdge Action;
	Action.NodeTemplate = NewObject<UEdNode_PSEEdge>(NodeA->GetGraph());
	Action.NodeTemplate->SetEdge(NewObject<UPuzzleSequencerEdge>(Action.NodeTemplate, Graph->EdgeType));
	UEdNode_PSEEdge* EdgeNode = Cast<UEdNode_PSEEdge>(Action.PerformAction(NodeA->GetGraph(), nullptr, InitPos, false));

	// Always create connections from node A to B, don't allow adding in reverse
	EdgeNode->CreateConnections(NodeA, NodeB);

	return true;
}

FConnectionDrawingPolicy* UAssetGraphSchema_PSE::CreateConnectionDrawingPolicy(int32 InBackLayerID, int32 InFrontLayerID, float InZoomFactor, const FSlateRect& InClippingRect, FSlateWindowElementList& InDrawElements, UEdGraph* InGraphObj) const
{
	return new FConnectionDrawingPolicy_PSE(InBackLayerID, InFrontLayerID, InZoomFactor, InClippingRect, InDrawElements, InGraphObj);
}

FLinearColor UAssetGraphSchema_PSE::GetPinTypeColor(const FEdGraphPinType& PinType) const
{
	return FColor::White;
}

void UAssetGraphSchema_PSE::BreakNodeLinks(UEdGraphNode& TargetNode) const
{
	const FScopedTransaction Transaction(NSLOCTEXT("UnrealEd", "GraphEd_BreakNodeLinks", "Break Node Links"));
	Super::BreakNodeLinks(TargetNode);
}

void UAssetGraphSchema_PSE::BreakPinLinks(UEdGraphPin& TargetPin, bool bSendsNodeNotifcation) const
{
	const FScopedTransaction Transaction(NSLOCTEXT("UnrealEd", "GraphEd_BreakPinLinks", "Break Pin Links"));
	Super::BreakPinLinks(TargetPin, bSendsNodeNotifcation);
}

void UAssetGraphSchema_PSE::BreakSinglePinLink(UEdGraphPin* SourcePin, UEdGraphPin* TargetPin) const
{
	const FScopedTransaction Transaction(NSLOCTEXT("UnrealEd", "GraphEd_BreakSinglePinLink", "Break Pin Link"));
	Super::BreakSinglePinLink(SourcePin, TargetPin);
}

UEdGraphPin* UAssetGraphSchema_PSE::DropPinOnNode(UEdGraphNode* InTargetNode, const FName& InSourcePinName, const FEdGraphPinType& InSourcePinType, EEdGraphPinDirection InSourcePinDirection) const
{
	UEdNode_PSENode* EdNode = Cast<UEdNode_PSENode>(InTargetNode);
	switch (InSourcePinDirection)
	{
	case EGPD_Input:
		return EdNode->GetOutputPin();
	case EGPD_Output:
		return EdNode->GetInputPin();
	default:
		return nullptr;
	}
}

bool UAssetGraphSchema_PSE::SupportsDropPinOnNode(UEdGraphNode* InTargetNode, const FEdGraphPinType& InSourcePinType, EEdGraphPinDirection InSourcePinDirection, FText& OutErrorMessage) const
{
	return Cast<UEdNode_PSENode>(InTargetNode) != nullptr;
}

bool UAssetGraphSchema_PSE::IsCacheVisualizationOutOfDate(int32 InVisualizationCacheID) const
{
	return CurrentCacheRefreshID != InVisualizationCacheID;
}

int32 UAssetGraphSchema_PSE::GetCurrentVisualizationCacheID() const
{
	return CurrentCacheRefreshID;
}

void UAssetGraphSchema_PSE::ForceVisualizationCacheClear() const
{
	++CurrentCacheRefreshID;
}

#undef LOCTEXT_NAMESPACE
