#pragma once
#include "EdGraphUtilities.h"

class PUZZLESEQUENCEREDITOR_API FPSENodeFactory : public FGraphPanelNodeFactory
{
	virtual TSharedPtr<class SGraphNode> CreateNode(UEdGraphNode* Node) const override;
};
