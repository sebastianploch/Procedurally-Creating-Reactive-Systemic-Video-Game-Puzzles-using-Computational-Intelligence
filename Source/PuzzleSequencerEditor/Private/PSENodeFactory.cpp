#include "PuzzleSequencerEditor/Public/PSENodeFactory.h"
#include "EdGraph/EdGraphNode.h"

#include "AssetEditor/SEdNode_PSEEdge.h"
#include "AssetEditor/EdNode_PSEEdge.h"
#include "AssetEditor/SEdNode_PSENode.h"
#include "AssetEditor/EdNode_PSENode.h"

TSharedPtr<SGraphNode> FPSENodeFactory::CreateNode(UEdGraphNode* Node) const
{
	if (UEdNode_PSENode* EdNode_GraphNode = Cast<UEdNode_PSENode>(Node))
	{
		return SNew(SEdNode_PSENode, EdNode_GraphNode);
	}
	else if (UEdNode_PSEEdge* EdNode_Edge = Cast<UEdNode_PSEEdge>(Node))
	{
		return SNew(SEdNode_PSEEdge, EdNode_Edge);
	}
	return nullptr;
}
