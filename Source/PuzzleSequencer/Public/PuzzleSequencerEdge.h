#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "PuzzleSequencerEdge.generated.h"

class UPuzzleSequencer;

UCLASS(Blueprintable)
class PUZZLESEQUENCER_API UPuzzleSequencerEdge : public UObject
{
	GENERATED_BODY()

public:
	UPuzzleSequencerEdge();
	virtual ~UPuzzleSequencerEdge() override = default;

	UPROPERTY(VisibleAnywhere, Category="PuzzleSequencerNode")
	UPuzzleSequencer* Graph{};
};
