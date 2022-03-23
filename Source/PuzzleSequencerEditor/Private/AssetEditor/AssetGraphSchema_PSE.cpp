#include "AssetEditor/AssetGraphSchema_PSE.h"


FAssetSchemaAction_PSE_NewNode::FAssetSchemaAction_PSE_NewNode(const FText& InNodeCategory, const FText& InMenuDesc, const FText& InToolTip, const int32 InGrouping)
{
}

UEdGraphNode* FAssetSchemaAction_PSE_NewNode::PerformAction(UEdGraph* ParentGraph, UEdGraphPin* FromPin, const FVector2D Location, bool bSelectNewNode)
{
	return FEdGraphSchemaAction::PerformAction(ParentGraph, FromPin, Location, bSelectNewNode);
}

void FAssetSchemaAction_PSE_NewNode::AddReferencedObjects(FReferenceCollector& Collector)
{
	FEdGraphSchemaAction::AddReferencedObjects(Collector);
}

FAssetSchemaAction_PSE_NewEdge::FAssetSchemaAction_PSE_NewEdge(const FText& InNodeCategory, const FText& InMenuDesc, const FText& InToolTip, const int32 InGrouping)
{
}

UEdGraphNode* FAssetSchemaAction_PSE_NewEdge::PerformAction(UEdGraph* ParentGraph, UEdGraphPin* FromPin, const FVector2D Location, bool bSelectNewNode)
{
	return FEdGraphSchemaAction::PerformAction(ParentGraph, FromPin, Location, bSelectNewNode);
}

void FAssetSchemaAction_PSE_NewEdge::AddReferencedObjects(FReferenceCollector& Collector)
{
	FEdGraphSchemaAction::AddReferencedObjects(Collector);
}

void UAssetGraphSchema_PSE::GetBreakLinkToSubMenuActions(UToolMenu* Menu, UEdGraphPin* InGraphPin)
{
}

EGraphType UAssetGraphSchema_PSE::GetGraphType(const UEdGraph* TestEdGraph) const
{
	return Super::GetGraphType(TestEdGraph);
}

void UAssetGraphSchema_PSE::GetGraphContextActions(FGraphContextMenuBuilder& ContextMenuBuilder) const
{
	Super::GetGraphContextActions(ContextMenuBuilder);
}

void UAssetGraphSchema_PSE::GetContextMenuActions(UToolMenu* Menu, UGraphNodeContextMenuContext* Context) const
{
	Super::GetContextMenuActions(Menu, Context);
}

const FPinConnectionResponse UAssetGraphSchema_PSE::CanCreateConnection(const UEdGraphPin* A, const UEdGraphPin* B) const
{
	return Super::CanCreateConnection(A, B);
}

bool UAssetGraphSchema_PSE::TryCreateConnection(UEdGraphPin* A, UEdGraphPin* B) const
{
	return Super::TryCreateConnection(A, B);
}

bool UAssetGraphSchema_PSE::CreateAutomaticConversionNodeAndConnections(UEdGraphPin* A, UEdGraphPin* B) const
{
	return Super::CreateAutomaticConversionNodeAndConnections(A, B);
}

FConnectionDrawingPolicy* UAssetGraphSchema_PSE::CreateConnectionDrawingPolicy(int32 InBackLayerID, int32 InFrontLayerID, float InZoomFactor, const FSlateRect& InClippingRect, FSlateWindowElementList& InDrawElements, UEdGraph* InGraphObj) const
{
	return Super::CreateConnectionDrawingPolicy(InBackLayerID, InFrontLayerID, InZoomFactor, InClippingRect, InDrawElements, InGraphObj);
}

FLinearColor UAssetGraphSchema_PSE::GetPinTypeColor(const FEdGraphPinType& PinType) const
{
	return Super::GetPinTypeColor(PinType);
}

void UAssetGraphSchema_PSE::BreakNodeLinks(UEdGraphNode& TargetNode) const
{
	Super::BreakNodeLinks(TargetNode);
}

void UAssetGraphSchema_PSE::BreakPinLinks(UEdGraphPin& TargetPin, bool bSendsNodeNotifcation) const
{
	Super::BreakPinLinks(TargetPin, bSendsNodeNotifcation);
}

void UAssetGraphSchema_PSE::BreakSinglePinLink(UEdGraphPin* SourcePin, UEdGraphPin* TargetPin) const
{
	Super::BreakSinglePinLink(SourcePin, TargetPin);
}

UEdGraphPin* UAssetGraphSchema_PSE::DropPinOnNode(UEdGraphNode* InTargetNode, const FName& InSourcePinName, const FEdGraphPinType& InSourcePinType, EEdGraphPinDirection InSourcePinDirection) const
{
	return Super::DropPinOnNode(InTargetNode, InSourcePinName, InSourcePinType, InSourcePinDirection);
}

bool UAssetGraphSchema_PSE::SupportsDropPinOnNode(UEdGraphNode* InTargetNode, const FEdGraphPinType& InSourcePinType, EEdGraphPinDirection InSourcePinDirection, FText& OutErrorMessage) const
{
	return Super::SupportsDropPinOnNode(InTargetNode, InSourcePinType, InSourcePinDirection, OutErrorMessage);
}

bool UAssetGraphSchema_PSE::IsCacheVisualizationOutOfDate(int32 InVisualizationCacheID) const
{
	return Super::IsCacheVisualizationOutOfDate(InVisualizationCacheID);
}

int32 UAssetGraphSchema_PSE::GetCurrentVisualizationCacheID() const
{
	return Super::GetCurrentVisualizationCacheID();
}

void UAssetGraphSchema_PSE::ForceVisualizationCacheClear() const
{
	Super::ForceVisualizationCacheClear();
}
