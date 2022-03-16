#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "PuzzleActor.generated.h"

UENUM(BlueprintType)
enum class EPuzzleType : uint8
{
	None = 0 UMETA(DisplayName="None"),
	Start UMETA(DisplayName="Start"),
	Intermediate UMETA(DisplayName="Intermediate"),
	Finish UMETA(DisplayName="Finish")
};

UCLASS()
class PUZZLESEQUENCER_API APuzzleActor final : public AActor
{
	GENERATED_BODY()

public:
	APuzzleActor();

	virtual void Tick(float DeltaTime) override;

protected:
	virtual void BeginPlay() override;

private:
	UPROPERTY(EditDefaultsOnly)
	EPuzzleType Type{EPuzzleType::None};
};
