#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "PuzzleSequencerNode.generated.h"

class UPuzzleSequencer;
class UPuzzleSequencerEdge;

UENUM(BlueprintType)
enum class ENodeLimit : uint8
{
	Unlimited,
	Limited
};

UCLASS()
class PUZZLESEQUENCER_API UPuzzleSequencerNode : public UObject
{
	GENERATED_BODY()

public:
	UPuzzleSequencerNode();
	virtual ~UPuzzleSequencerNode() override = default;

	UPROPERTY(VisibleDefaultsOnly, Category="PuzzleSequencerNode")
	UPuzzleSequencer* Graph;

	UPROPERTY(BlueprintReadOnly, Category="PuzzleSequencerNode")
	TArray<UPuzzleSequencerNode*> ParentNodes;

	UPROPERTY(BlueprintReadOnly, Category="PuzzleSequencerNode")
	TArray<UPuzzleSequencerNode*> ChildrenNodes;

	UPROPERTY(BlueprintReadOnly, Category="PuzzleSequencerNode")
	TMap<UPuzzleSequencerNode*, UPuzzleSequencerEdge*> Edges;

	UFUNCTION(BlueprintCallable, Category="PuzzleSequencerNode")
	virtual UPuzzleSequencerEdge* GetEdge(UPuzzleSequencerNode* InChildNode);

	UFUNCTION(BlueprintCallable, Category="PuzzleSequencerNode")
	bool IsLeafNode() const;

	UFUNCTION(BlueprintCallable, Category="PuzzleSequencerNode")
	UPuzzleSequencer* GetGraph() const;

	UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category="PuzzleSequencerNode")
	FText GetDescription() const;
	virtual FText GetDescription_Implementation() const;

	//////////////////////////////////////////////////////////////////////////
#if WITH_EDITORONLY_DATA
	UPROPERTY(EditDefaultsOnly, Category = "PuzzleSequencerNode")
	FText NodeTitle;

	UPROPERTY(VisibleDefaultsOnly, Category = "PuzzleSequencerNode")
	TSubclassOf<UPuzzleSequencer> CompatibleGraphType;

	UPROPERTY(EditDefaultsOnly, Category = "PuzzleSequencerNode")
	FLinearColor BackgroundColor;

	UPROPERTY(EditDefaultsOnly, Category = "PuzzleSequencerNode")
	FText ContextMenuName;

	UPROPERTY(EditDefaultsOnly, Category = "PuzzleSequencerNode")
	ENodeLimit ParentLimitType;

	UPROPERTY(EditDefaultsOnly, Category = "PuzzleSequencerNode", meta = (ClampMin = "0", EditCondition = "ParentLimitType == ENodeLimit::Limited", EditConditionHides))
	int32 ParentLimit;

	UPROPERTY(EditDefaultsOnly, Category = "PuzzleSequencerNode")
	ENodeLimit ChildrenLimitType;

	UPROPERTY(EditDefaultsOnly, Category = "PuzzleSequencerNode", meta = (ClampMin = "0", EditCondition = "ChildrenLimitType == ENodeLimit::Limited", EditConditionHides))
	int32 ChildrenLimit;

#endif

#if WITH_EDITOR
	virtual bool IsNameEditable() const;

	virtual FLinearColor GetBackgroundColor() const;

	virtual FText GetNodeTitle() const;

	virtual void SetNodeTitle(const FText& NewTitle);

	virtual bool CanCreateConnection(UPuzzleSequencerNode* Other, FText& ErrorMessage);

	virtual bool CanCreateConnectionTo(UPuzzleSequencerNode* Other, int32 NumberOfChildrenNodes, FText& ErrorMessage);
	virtual bool CanCreateConnectionFrom(UPuzzleSequencerNode* Other, int32 NumberOfParentNodes, FText& ErrorMessage);
#endif
};
