#pragma once

#include "CoreMinimal.h"
#include "PuzzleSequencer.h"
#include "PuzzleSequencerNode.h"
#include "PuzzleSequencerEdge.h"
#include "AssetGraphSchema_PSE.generated.h"

class UEdNode_PSENode;
class UEdNode_PSEEdge;

#pragma region NewNode
USTRUCT()
struct PUZZLESEQUENCEREDITOR_API FAssetSchemaAction_PSE_NewNode : public FEdGraphSchemaAction
{
	GENERATED_BODY()

	FAssetSchemaAction_PSE_NewNode() = default;

	FAssetSchemaAction_PSE_NewNode(const FText& InNodeCategory, const FText& InMenuDesc, const FText& InToolTip, const int32 InGrouping);

	virtual UEdGraphNode* PerformAction(UEdGraph* ParentGraph, UEdGraphPin* FromPin, const FVector2D Location, bool bSelectNewNode) override;
	virtual void AddReferencedObjects(FReferenceCollector& Collector) override;

public:
	UPROPERTY()
	UEdNode_PSENode* NodeTemplate{nullptr};
};
#pragma endregion NewNode

#pragma region NewEdge
USTRUCT()
struct PUZZLESEQUENCEREDITOR_API FAssetSchemaAction_PSE_NewEdge : public FEdGraphSchemaAction
{
	GENERATED_BODY()

	FAssetSchemaAction_PSE_NewEdge() = default;

	FAssetSchemaAction_PSE_NewEdge(const FText& InNodeCategory, const FText& InMenuDesc, const FText& InToolTip, const int32 InGrouping);

	virtual UEdGraphNode* PerformAction(UEdGraph* ParentGraph, UEdGraphPin* FromPin, const FVector2D Location, bool bSelectNewNode) override;
	virtual void AddReferencedObjects(FReferenceCollector& Collector) override;

public:
	UPROPERTY()
	UEdNode_PSEEdge* NodeTemplate{nullptr};
};
#pragma endregion NewEdge

UCLASS(MinimalAPI)
class UAssetGraphSchema_PSE : public UEdGraphSchema
{
	GENERATED_BODY()

public:
	void GetBreakLinkToSubMenuActions(class UToolMenu* Menu, class UEdGraphPin* InGraphPin);

	virtual EGraphType GetGraphType(const UEdGraph* TestEdGraph) const override;

	virtual void GetGraphContextActions(FGraphContextMenuBuilder& ContextMenuBuilder) const override;
	virtual void GetContextMenuActions(UToolMenu* Menu, UGraphNodeContextMenuContext* Context) const override;

	virtual const FPinConnectionResponse CanCreateConnection(const UEdGraphPin* A, const UEdGraphPin* B) const override;
	virtual bool TryCreateConnection(UEdGraphPin* A, UEdGraphPin* B) const override;
	virtual bool CreateAutomaticConversionNodeAndConnections(UEdGraphPin* A, UEdGraphPin* B) const override;
	virtual class FConnectionDrawingPolicy* CreateConnectionDrawingPolicy(int32 InBackLayerID, int32 InFrontLayerID, float InZoomFactor, const FSlateRect& InClippingRect, FSlateWindowElementList& InDrawElements, UEdGraph* InGraphObj) const override;

	virtual FLinearColor GetPinTypeColor(const FEdGraphPinType& PinType) const override;

	virtual void BreakNodeLinks(UEdGraphNode& TargetNode) const override;
	virtual void BreakPinLinks(UEdGraphPin& TargetPin, bool bSendsNodeNotifcation) const override;
	virtual void BreakSinglePinLink(UEdGraphPin* SourcePin, UEdGraphPin* TargetPin) const override;

	virtual UEdGraphPin* DropPinOnNode(UEdGraphNode* InTargetNode, const FName& InSourcePinName, const FEdGraphPinType& InSourcePinType, EEdGraphPinDirection InSourcePinDirection) const override;
	virtual bool SupportsDropPinOnNode(UEdGraphNode* InTargetNode, const FEdGraphPinType& InSourcePinType, EEdGraphPinDirection InSourcePinDirection, FText& OutErrorMessage) const override;

	virtual bool IsCacheVisualizationOutOfDate(int32 InVisualizationCacheID) const override;
	virtual int32 GetCurrentVisualizationCacheID() const override;
	virtual void ForceVisualizationCacheClear() const override;

private:
	inline static int32 CurrentCacheRefreshID{0};
};
