#pragma once

#include "CoreMinimal.h"
#include "PuzzleSequencerNode.h"
#include "PuzzleSequencerEdge.generated.h"

class UPuzzleSequencer;

UCLASS(Blueprintable)
class PUZZLESEQUENCER_API UPuzzleSequencerEdge : public UObject
{
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, Category="PuzzleSequencerNode")
	UPuzzleSequencer* Graph{nullptr};

	UPROPERTY(BlueprintReadOnly, Category="PuzzleSequencerEdge")
	UPuzzleSequencerNode* StartNode{nullptr};

	UPROPERTY(BlueprintReadOnly, Category="PuzzleSequencerEdge")
	UPuzzleSequencerNode* EndNode{nullptr};

	UFUNCTION(BlueprintPure, Category="PuzzleSequencerEdge")
	UPuzzleSequencer* GetGraph() const;

#pragma region Editor
#if WITH_EDITOR
public:
	virtual FText GetNodeTitle() const { return NodeTitle; }
	FLinearColor GetEdgeColour() const { return EdgeColour; }

	virtual void SetNodeTitle(const FText& InNewTitle);
#endif

#if WITH_EDITORONLY_DATA
public:
	UPROPERTY(EditDefaultsOnly, Category="PuzzleSequencerNode_Editor")
	bool bShouldDrawTitle{false};

	UPROPERTY(EditDefaultsOnly, Category="PuzzleSequencerNode_Editor")
	FText NodeTitle{FText::FromName(NAME_Name)};

	UPROPERTY(EditDefaultsOnly, Category="PuzzleSequencerNode_Editor")
	FLinearColor EdgeColour{0.9f, 0.9f, 0.9f, 1.f};
#endif
#pragma endregion Editor
};
