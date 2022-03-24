#pragma once

#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "PuzzleSequencer.h"
#include "PSEFactory.generated.h"

UCLASS()
class PUZZLESEQUENCEREDITOR_API UPSEFactory : public UFactory
{
	GENERATED_BODY()

public:
	UPSEFactory();
	
	virtual bool ConfigureProperties() override;
	virtual UObject* FactoryCreateNew(UClass* InClass, UObject* InParent, FName InName, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn) override;

public:
	UPROPERTY(EditAnywhere, Category="DataAsset")
	TSubclassOf<UPuzzleSequencer> PuzzleSequencerClass{UPuzzleSequencer::StaticClass()};
};
