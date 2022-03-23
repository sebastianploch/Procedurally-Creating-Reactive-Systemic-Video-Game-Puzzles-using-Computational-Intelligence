#include "AssetEditor/EdGraph_PSE.h"
#include "PuzzleSequencer.h"
#include "AssetEditor/EdNode_PSENode.h"
#include "AssetEditor/EdNode_PSEEdge.h"

void UEdGraph_PSE::RebuildGraph()
{
	UPuzzleSequencer* Graph = GetGraph();

	Clear();

	for (int i = 0; i < Nodes.Num(); ++i)
	{
		if (UEdNode_PSENode* EdNode = Cast<UEdNode_PSENode>(Nodes[i]))
		{
			if (EdNode->Node == nullptr)
			{
				continue;
			}

			UPuzzleSequencerNode* GenericGraphNode = EdNode->Node;

			NodeMap.Add(GenericGraphNode, EdNode);

			Graph->AllNodes.Add(GenericGraphNode);

			for (int PinIdx = 0; PinIdx < EdNode->Pins.Num(); ++PinIdx)
			{
				UEdGraphPin* Pin = EdNode->Pins[PinIdx];

				if (Pin->Direction != EEdGraphPinDirection::EGPD_Output)
				{
					continue;
				}

				for (int LinkToIdx = 0; LinkToIdx < Pin->LinkedTo.Num(); ++LinkToIdx)
				{
					UPuzzleSequencerNode* ChildNode = nullptr;
					if (UEdNode_PSENode* EdNode_Child = Cast<UEdNode_PSENode>(Pin->LinkedTo[LinkToIdx]->GetOwningNode()))
					{
						ChildNode = EdNode_Child->Node;
					}
					else if (UEdNode_PSEEdge* EdNode_Edge = Cast<UEdNode_PSEEdge>(Pin->LinkedTo[LinkToIdx]->GetOwningNode()))
					{
						UEdNode_PSENode* Child = EdNode_Edge->GetEndNode();;
						if (Child != nullptr)
						{
							ChildNode = Child->Node;
						}
					}

					if (ChildNode != nullptr)
					{
						GenericGraphNode->ChildrenNodes.Add(ChildNode);

						ChildNode->ParentNodes.Add(GenericGraphNode);
					}
					else
					{
						//LOG_ERROR(TEXT("UEdGraph_GenericGraph::RebuildGenericGraph can't find child node"));
					}
				}
			}
		}
		else if (UEdNode_PSEEdge* EdgeNode = Cast<UEdNode_PSEEdge>(Nodes[i]))
		{
			UEdNode_PSENode* StartNode = EdgeNode->GetStartNode();
			UEdNode_PSENode* EndNode = EdgeNode->GetEndNode();
			UPuzzleSequencerEdge* Edge = EdgeNode->Edge;

			if (StartNode == nullptr || EndNode == nullptr || Edge == nullptr)
			{
				//LOG_ERROR(TEXT("UEdGraph_GenericGraph::RebuildGenericGraph add edge failed."));
				continue;
			}

			EdgeMap.Add(Edge, EdgeNode);

			Edge->Graph = Graph;
			Edge->Rename(nullptr, Graph, REN_DontCreateRedirectors | REN_DoNotDirty);
			Edge->StartNode = StartNode->Node;
			Edge->EndNode = EndNode->Node;
			Edge->StartNode->Edges.Add(Edge->EndNode, Edge);
		}
	}

	for (int i = 0; i < Graph->AllNodes.Num(); ++i)
	{
		UPuzzleSequencerNode* Node = Graph->AllNodes[i];
		if (Node->ParentNodes.Num() == 0)
		{
			Graph->RootNodes.Add(Node);

			SortNodes(Node);
		}

		Node->Graph = Graph;
		Node->Rename(nullptr, Graph, REN_DontCreateRedirectors | REN_DoNotDirty);
	}

	Graph->RootNodes.Sort([&](const UPuzzleSequencerNode& L, const UPuzzleSequencerNode& R)
	{
		UEdNode_PSENode* EdNode_LNode = NodeMap[&L];
		UEdNode_PSENode* EdNode_RNode = NodeMap[&R];
		return EdNode_LNode->NodePosX < EdNode_RNode->NodePosX;
	});
}

UPuzzleSequencer* UEdGraph_PSE::GetGraph()
{
	return CastChecked<UPuzzleSequencer>(GetOuter());
}

bool UEdGraph_PSE::Modify(bool bAlwaysMarkDirty)
{
	bool Rtn = Super::Modify(bAlwaysMarkDirty);

	GetGraph()->Modify();

	for (int32 i = 0; i < Nodes.Num(); ++i)
	{
		Nodes[i]->Modify();
	}

	return Rtn;
}

void UEdGraph_PSE::PostEditUndo()
{
	Super::PostEditUndo();

	NotifyGraphChanged();
}

void UEdGraph_PSE::Clear()
{
	UPuzzleSequencer* Graph = GetGraph();

	Graph->ClearGraph();
	NodeMap.Reset();
	EdgeMap.Reset();

	for (int i = 0; i < Nodes.Num(); ++i)
	{
		if (UEdNode_PSENode* EdNode = Cast<UEdNode_PSENode>(Nodes[i]))
		{
			UPuzzleSequencerNode* GenericGraphNode = EdNode->Node;
			if (GenericGraphNode)
			{
				GenericGraphNode->ParentNodes.Reset();
				GenericGraphNode->ChildrenNodes.Reset();
				GenericGraphNode->Edges.Reset();
			}
		}
	}
}

void UEdGraph_PSE::SortNodes(UPuzzleSequencerNode* InRootNode)
{
	int Level = 0;
	TArray<UPuzzleSequencerNode*> CurrLevelNodes = {InRootNode};
	TArray<UPuzzleSequencerNode*> NextLevelNodes;

	while (CurrLevelNodes.Num() != 0)
	{
		int32 LevelWidth = 0;
		for (int i = 0; i < CurrLevelNodes.Num(); ++i)
		{
			UPuzzleSequencerNode* Node = CurrLevelNodes[i];

			auto Comp = [&](const UPuzzleSequencerNode& L, const UPuzzleSequencerNode& R)
			{
				UEdNode_PSENode* EdNode_LNode = NodeMap[&L];
				UEdNode_PSENode* EdNode_RNode = NodeMap[&R];
				return EdNode_LNode->NodePosX < EdNode_RNode->NodePosX;
			};

			Node->ChildrenNodes.Sort(Comp);
			Node->ParentNodes.Sort(Comp);

			for (int j = 0; j < Node->ChildrenNodes.Num(); ++j)
			{
				NextLevelNodes.Add(Node->ChildrenNodes[j]);
			}
		}

		CurrLevelNodes = NextLevelNodes;
		NextLevelNodes.Reset();
		++Level;
	}
}
