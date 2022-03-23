#pragma once

#include "CoreMinimal.h"
#include "Styling/SlateColor.h"
#include "Widgets/DeclarativeSyntaxSupport.h"
#include "Widgets/SWidget.h"
#include "SNodePanel.h"
#include "SGraphNode.h"

class SToolTip;
class UEdNode_PSEEdge;

class PUZZLESEQUENCEREDITOR_API SEdNode_PSEEdge : public SGraphNode
{
public:
	SLATE_BEGIN_ARGS(SEdNode_PSEEdge)
		{
		}

	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs, UEdNode_PSEEdge* InNode);

	virtual bool RequiresSecondPassLayout() const override;
	virtual void PerformSecondPassLayout(const TMap<UObject*, TSharedRef<SNode>>& InNodeToWidgetLookup) const override;

	virtual void UpdateGraphNode() override;

	void PositionBetweenTwoNodesWithOffset(const FGeometry& InStartGeometry, const FGeometry& InEndGeometry, int32 InNodeIndex, int32 InMaxNodes) const;

	void OnNameTextCommited(const FText& InText, ETextCommit::Type InCommitInfo);

protected:
	FSlateColor GetEdgeColour() const;

	const FSlateBrush* GetEdgeImage() const;

	EVisibility GetEdgeImageVisibility() const;
	EVisibility GetEdgeTitleVisibility() const;

private:
	TSharedPtr<STextEntryPopup> TextEntryWidget{};
};
