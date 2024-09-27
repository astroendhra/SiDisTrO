from aptos_sdk.account import Account
from aptos_sdk.async_client import RestClient
from aptos_sdk.transactions import EntryFunction, TransactionArgument, TransactionPayload
from aptos_sdk.type_tag import StructTag, TypeTag
import logging

class AptosIntegration:
    def __init__(self, node_url, private_key):
        self.client = RestClient(node_url)
        self.account = Account.load_key(private_key)

    def store_checkpoint(self, epoch, model_hash):
        payload = EntryFunction.natural(
            "0x8ff26600cf44824ab062c8deaa8b6d3d763f003a223cc56762dd108c580db384::training",  # Replace with your actual module address
            "store_checkpoint",
            [],
            [
                TransactionArgument(epoch, TypeTag.U64()),
                TransactionArgument(bytes.fromhex(model_hash), TypeTag.vector(TypeTag.u8())),
            ],
        )
        
        signed_transaction = self.client.create_single_signer_bcs_transaction(
            self.account, TransactionPayload(payload)
        )
        
        self.client.submit_bcs_transaction(signed_transaction)
        logging.info(f"Stored checkpoint for epoch {epoch} with hash {model_hash}")

    def verify_participation(self, participant_address):
        payload = EntryFunction.natural(
            "0x8ff26600cf44824ab062c8deaa8b6d3d763f003a223cc56762dd108c580db384::training",  # Replace with your actual module address
            "verify_participation",
            [],
            [TransactionArgument(participant_address, TypeTag.address())],
        )
        
        signed_transaction = self.client.create_single_signer_bcs_transaction(
            self.account, TransactionPayload(payload)
        )
        
        self.client.submit_bcs_transaction(signed_transaction)
        logging.info(f"Verified participation for address {participant_address}")

    def distribute_rewards(self):
        payload = EntryFunction.natural(
            "0x8ff26600cf44824ab062c8deaa8b6d3d763f003a223cc56762dd108c580db384::training",  # Replace with your actual module address
            "distribute_rewards",
            [],
            [],
        )
        
        signed_transaction = self.client.create_single_signer_bcs_transaction(
            self.account, TransactionPayload(payload)
        )
        
        self.client.submit_bcs_transaction(signed_transaction)
        logging.info("Distributed rewards to participants")