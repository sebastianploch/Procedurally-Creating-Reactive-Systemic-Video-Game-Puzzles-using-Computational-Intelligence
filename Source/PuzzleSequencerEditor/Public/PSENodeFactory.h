#pragma once
#include "EdGraphUtilities.h"
#include "EdGraph/EdGraphNode.h"

class PUZZLESEQUENCEREDITOR_API FPSENodeFactory : public FGraphPanelNodeFactory
{
	virtual TSharedPtr<class SGraphNode> CreateNode(UEdGraphNode* Node) const override;
};
