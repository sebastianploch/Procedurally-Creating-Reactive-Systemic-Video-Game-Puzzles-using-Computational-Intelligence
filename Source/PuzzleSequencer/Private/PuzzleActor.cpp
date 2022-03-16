#include "PuzzleActor.h"

APuzzleActor::APuzzleActor()
{
	PrimaryActorTick.bCanEverTick = true;
}

void APuzzleActor::BeginPlay()
{
	Super::BeginPlay();
}

void APuzzleActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}
