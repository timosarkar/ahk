module memecoin::samplecoin {
  use sui::coin::{Self, Coin, TreasuryCap};

  const AMOUNT: u64 = 600000000000000000; // 600M 60% of the supply

  public struct SAMPLECOIN has drop {}

  fun init(witness: SAMPLECOIN, ctx: &mut TxContext) {
      let (mut treasury, metadata) = coin::create_currency<SAMPLECOIN>(
            witness, 
            9,
            b"SAMPLECOIN",
            b"just a simple memecoin",
            b"just a simple memecoin lol.",
            option::none(),
            ctx
        );

      let sender = tx_context::sender(ctx);  

      coin::mint_and_transfer(&mut treasury, AMOUNT, sender, ctx);

      transfer::public_transfer(treasury, sender);
      transfer::public_freeze_object(metadata);
  }

  entry fun mint(cap: &mut TreasuryCap<SAMPLECOIN>, value: u64, sender: address, ctx: &mut TxContext,) {
     coin::mint_and_transfer(cap, value, sender, ctx);
  }

  public entry fun transfer(c: Coin<SAMPLECOIN>, recipient: address) {
    transfer::public_transfer(c, recipient);
  }
}
