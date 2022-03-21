#include "PuzzleSequencerEditor/Public/PSENodeFactory.h"
#include "EdGraph/EdGraphNode.h"

#include "AssetEditor/SEdNode_PSEEdge.h"
#include "AssetEditor/EdNode_PSEEdge.h"
#include "AssetEditor/SEdNode_PSENode.h"
#include "AssetEditor/EdNode_PSENode.h"

TSharedPtr<SGraphNode> FPSENodeFactory::CreateNode(UEdGraphNode* Node) const
{
	return nullptr;
}
