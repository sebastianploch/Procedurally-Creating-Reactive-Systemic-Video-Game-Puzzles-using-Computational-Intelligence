#pragma once

#include "CoreMinimal.h"
#include "EdGraph/EdGraphNode.h"
#include "EdNode_PSEEdge.generated.h"

UCLASS(MinimalAPI)
class PUZZLESEQUENCEREDITOR_API UEdNode_PSEEdge : public UEdGraphNode
{
	GENERATED_BODY()

public:
	UEdNode_PSEEdge();

	UPROPERTY()
	class UEdGraph* Grap;
};
