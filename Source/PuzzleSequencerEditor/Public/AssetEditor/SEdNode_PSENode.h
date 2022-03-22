#pragma once

#include "CoreMinimal.h"
#include "SGraphNode.h"

class UEdNode_PSENode;

class PUZZLESEQUENCEREDITOR_API SEdNode_PSENode : public SGraphNode
{
public:
	SLATE_BEGIN_ARGS(SEdNode_PSENode)
		{
		}

	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs, UEdNode_PSENode* InNode);

	virtual void UpdateGraphNode() override;
	virtual void CreatePinWidgets() override;
	virtual void AddPin(const TSharedRef<SGraphPin>& PinToAdd) override;
	virtual bool IsNameReadOnly() const override;

	void OnNameTextCommited(const FText& InText, ETextCommit::Type InCommitInfo);

	virtual FSlateColor GetBorderBackgroundColour() const;
	virtual FSlateColor GetBackgroundColour() const;

	virtual EVisibility GetDragOverMarkerVisibility() const;

	virtual const FSlateBrush* GetNameIcon() const;
};
