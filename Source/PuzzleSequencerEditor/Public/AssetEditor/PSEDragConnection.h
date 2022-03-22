#pragma once

#include "CoreMinimal.h"
#include "Input/DragAndDrop.h"
#include "Input/Reply.h"
#include "Widgets/SWidget.h"
#include "SGraphPin.h"
#include "GraphEditorDragDropAction.h"

class SGraphPanel;
class UEdGraph;

class FPSEDragConnection : public FGraphEditorDragDropAction
{
public:
	DRAG_DROP_OPERATOR_TYPE(FPSEDragConnection, FGraphEditorDragDropAction)

	using FDraggedPinTable = TArray<FGraphPinHandle>;
	static TSharedRef<FPSEDragConnection> New(const TSharedRef<SGraphPanel>& InGraphPanel, const FDraggedPinTable& InDraggedPins);

	virtual void OnDrop(bool bDropWasHandled, const FPointerEvent& MouseEvent) override;

	virtual void HoverTargetChanged() override;
	virtual FReply DroppedOnPin(FVector2D ScreenPosition, FVector2D GraphPosition) override;
	virtual FReply DroppedOnNode(FVector2D ScreenPosition, FVector2D GraphPosition) override;
	virtual FReply DroppedOnPanel(const TSharedRef<SWidget>& Panel, FVector2D ScreenPosition, FVector2D GraphPosition, UEdGraph& Graph) override;
	virtual void OnDragged(const FDragDropEvent& DragDropEvent) override;

	virtual void ValidateGraphPinList(TArray<UEdGraphPin*>& OutValidPins);

protected:
	FPSEDragConnection(const TSharedRef<SGraphPanel>& InGraphPanel, const FDraggedPinTable& InDraggedPins);

protected:
	TSharedPtr<SGraphPanel> GraphPanel{};
	FDraggedPinTable DraggingPins{};

	FVector2D DecoratorAdjust{};
};
