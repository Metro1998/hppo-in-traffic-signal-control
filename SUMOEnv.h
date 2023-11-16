#ifndef SUMOENV_H
#define SUMOENV_H

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace sumoenv {


class SumoEnvFns {
    public:
        static decltype(auto) DefaultConfig(){
            return MakeDict("num_agents"_.Bind(10))

        }

        template <typename Config>
        static decltype(auto) StateSpec(const Config& conf){
            return MakeDict("obs"_.bind(
                Spec<int>({})
            ))
        }






}





}






































#endif // SUMOENV_H