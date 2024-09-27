module distributed_training {
    use std::vector;
    use aptos_framework::coin;
    use aptos_framework::aptos_coin::AptosCoin;

    struct TrainingParticipant has key {
        address: address,
        checkpoints: vector<u64>,
        reward: u64,
    }

    struct TrainingState has key {
        participants: vector<address>,
        current_epoch: u64,
        total_reward: u64,
    }

    public fun store_checkpoint(account: &signer, epoch: u64, model_hash: vector<u8>) acquires TrainingState {
        let sender = std::signer::address_of(account);
        let training_state = borrow_global_mut<TrainingState>(@distributed_training);
        
        if (!vector::contains(&training_state.participants, &sender)) {
            vector::push_back(&mut training_state.participants, sender);
        }

        let participant = borrow_global_mut<TrainingParticipant>(sender);
        vector::push_back(&mut participant.checkpoints, epoch);
        
        // Update current epoch
        if (epoch > training_state.current_epoch) {
            training_state.current_epoch = epoch;
        }
    }

    public fun verify_participation(account: &signer, participant_address: address) acquires TrainingParticipant {
        let participant = borrow_global_mut<TrainingParticipant>(participant_address);
        participant.reward = participant.reward + 1;
    }

    public fun distribute_rewards(account: &signer) acquires TrainingState, TrainingParticipant {
        let training_state = borrow_global_mut<TrainingState>(@distributed_training);
        let total_participants = vector::length(&training_state.participants);
        
        let i = 0;
        while (i < total_participants) {
            let participant_address = *vector::borrow(&training_state.participants, i);
            let participant = borrow_global_mut<TrainingParticipant>(participant_address);
            
            let reward_amount = (participant.reward * training_state.total_reward) / total_participants;
            coin::transfer<AptosCoin>(account, participant_address, reward_amount);
            
            participant.reward = 0;
            i = i + 1;
        }
        
        training_state.total_reward = 0;
    }
}